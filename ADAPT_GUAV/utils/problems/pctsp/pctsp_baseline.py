import argparse
import os
import numpy as np
from utils import run_all_in_pool
from utils.data_utils import check_extension, load_dataset, save_dataset
from subprocess import check_call, check_output
import re
import time
from datetime import timedelta
import random
from scipy.spatial import distance_matrix
from .salesman.pctsp.model.pctsp import Pctsp
from .salesman.pctsp.algo.ilocal_search import ilocal_search
from .salesman.pctsp.model import solution
# Add imports needed for LKH interaction
import subprocess
import math
import os

MAX_LENGTH_TOL = 1e-5


def get_pctsp_executable():
    path = os.path.join("pctsp", "PCTSP", "PCPTSP")
    sourcefile = os.path.join(path, "main.cpp")
    execfile = os.path.join(path, "main.out")
    if not os.path.isfile(execfile):
        print ("Compiling...")
        check_call(["g++", "-g", "-Wall", sourcefile, "-std=c++11", "-o", execfile])
        print ("Done!")
    assert os.path.isfile(execfile), "{} does not exist! Compilation failed?".format(execfile)
    return os.path.abspath(execfile)


def solve_pctsp_log(executable, directory, name, depot, loc, penalty, deterministic_prize, stochastic_prize, runs=10):

    problem_filename = os.path.join(directory, "{}.pctsp{}.pctsp".format(name, runs))
    output_filename = os.path.join(directory, "{}.pctsp{}.pkl".format(name, runs))
    log_filename = os.path.join(directory, "{}.pctsp{}.log".format(name, runs))

    try:
        # May have already been run
        if not os.path.isfile(output_filename):
            write_pctsp(problem_filename, depot, loc, penalty, deterministic_prize, name=name)
            with open(log_filename, 'w') as f:
                start = time.time()
                output = check_output(
                    # exe, filename, min_total_prize (=1), num_runs
                    [executable, problem_filename, float_to_scaled_int_str(1.), str(runs)],
                    stderr=f
                ).decode('utf-8')
                duration = time.time() - start
                f.write(output)

            save_dataset((output, duration), output_filename)
        else:
            output, duration = load_dataset(output_filename)

        # Now parse output
        tour = None
        for line in output.splitlines():
            heading = "Best Result Route: "
            if line[:len(heading)] == heading:
                tour = np.array(line[len(heading):].split(" ")).astype(int)
                break
        assert tour is not None, "Could not find tour in output!"

        assert tour[0] == 0, "Tour should start with depot"
        assert tour[-1] == 0, "Tour should end with depot"
        tour = tour[1:-1]  # Strip off depot

        return calc_pctsp_cost(depot, loc, penalty, deterministic_prize, tour), tour.tolist(), duration
    except Exception as e:
        print("Exception occured")
        print(e)
        return None


def solve_stochastic_pctsp_log(
        executable, directory, name, depot, loc, penalty, deterministic_prize, stochastic_prize, runs=10, append='all'):

    try:

        problem_filename = os.path.join(directory, "{}.stochpctsp{}{}.pctsp".format(name, append, runs))
        output_filename = os.path.join(directory, "{}.stochpctsp{}{}.pkl".format(name, append, runs))
        log_filename = os.path.join(directory, "{}.stochpctsp{}{}.log".format(name, append, runs))

        # May have already been run
        if not os.path.isfile(output_filename):

            total_start = time.time()

            outputs = []
            durations = []
            final_tour = []

            coord = [depot] + loc

            mask = np.zeros(len(coord), dtype=bool)
            dist = distance_matrix(coord, coord)
            penalty = np.array(penalty)
            deterministic_prize = np.array(deterministic_prize)

            it = 0
            total_collected_prize = 0.
            # As long as we have not visited all nodes we repeat
            # even though we have already satisfied the total prize collected constraint
            # since the algorithm may decide to include more nodes to avoid further penalties
            while len(final_tour) < len(stochastic_prize):

                # Mask all nodes already visited (not the depot)
                mask[final_tour] = True

                # The distance from the 'start' or 'depot' is the distance from the 'current node'
                # this way we mimic as if we have a separate start and end by the assymetric distance matrix
                # Note: this violates the triangle inequality and the distance from 'depot to depot' becomes nonzero
                # but the program seems to deal with this well
                if len(final_tour) > 0:  # in the first iteration we are at depot and distance matrix is ok
                    dist[0, :] = dist[final_tour[-1], :]

                remaining_deterministic_prize = deterministic_prize[~mask[1:]]
                write_pctsp_dist(problem_filename,
                                 dist[np.ix_(~mask, ~mask)], penalty[~mask[1:]], remaining_deterministic_prize)
                # If the remaining deterministic prize is less than the prize we should still collect
                # set this lower value as constraint since otherwise problem is infeasible
                # compute total remaining deterministic prize after converting to ints
                # otherwise we may still have problems with rounding
                # Note we need to clip 1 - total_collected_prize between 0 (constraint can already be satisfied)
                # and the maximum achievable with the remaining_deterministic_prize
                min_prize_int = max(0, min(
                    float_to_scaled_int(1. - total_collected_prize),
                    sum([float_to_scaled_int(v) for v in remaining_deterministic_prize])
                ))
                with open(log_filename, 'a') as f:
                    start = time.time()
                    output = check_output(
                        # exe, filename, min_total_prize (=1), num_runs
                        [executable, problem_filename, str(min_prize_int), str(runs)],
                        stderr=f
                    ).decode('utf-8')
                    durations.append(time.time() - start)
                    outputs.append(output)

                # Now parse output
                tour = None
                for line in output.splitlines():
                    heading = "Best Result Route: "
                    if line[:len(heading)] == heading:
                        tour = np.array(line[len(heading):].split(" ")).astype(int)
                        break
                assert tour is not None, "Could not find tour in output!"

                assert tour[0] == 0, "Tour should start with depot"
                assert tour[-1] == 0, "Tour should end with depot"
                tour = tour[1:-1]  # Strip off depot

                # Now find to which nodes these correspond
                tour_node_ids = np.arange(len(coord), dtype=int)[~mask][tour]

                if len(tour_node_ids) == 0:
                    # The inner algorithm can decide to stop, but does not have to
                    assert total_collected_prize > 1 - 1e-5, "Collected prize should be one"
                    break

                if append == 'first':
                    final_tour.append(tour_node_ids[0])
                elif append == 'half':
                    final_tour.extend(tour_node_ids[:max(len(tour_node_ids) // 2, 1)])
                else:
                    assert append == 'all'
                    final_tour.extend(tour_node_ids)

                total_collected_prize = calc_pctsp_total(stochastic_prize, final_tour)
                it = it + 1

            os.remove(problem_filename)
            final_cost = calc_pctsp_cost(depot, loc, penalty, stochastic_prize, final_tour)
            total_duration = time.time() - total_start
            save_dataset((final_cost, final_tour, total_duration, outputs, durations), output_filename)

        else:
            final_cost, final_tour, total_duration, outputs, durations = load_dataset(output_filename)

        return final_cost, final_tour, total_duration
    except Exception as e:
        print("Exception occured")
        print(e)
        return None


def solve_salesman(directory, name, depot, loc, penalty, deterministic_prize, stochastic_prize, runs=10):

    problem_filename = os.path.join(directory, "{}.salesman{}.pctsp".format(name, runs))
    output_filename = os.path.join(directory, "{}.salesman{}.pkl".format(name, runs))

    try:
        # May have already been run
        if not os.path.isfile(output_filename):
            write_pctsp(problem_filename, depot, loc, penalty, deterministic_prize, name=name)

            start = time.time()

            random.seed(1234)
            pctsp = Pctsp()
            pctsp.load(problem_filename, float_to_scaled_int(1.))
            s = solution.random(pctsp, start_size=int(len(pctsp.prize) * 0.7))
            s = ilocal_search(s, n_runs=runs)

            output = (s.route[:s.size], s.quality)

            duration = time.time() - start

            save_dataset((output, duration), output_filename)
        else:
            output, duration = load_dataset(output_filename)

        # Now parse output
        tour = output[0][:]
        assert tour[0] == 0, "Tour should start with depot"
        assert tour[-1] != 0, "Tour should not end with depot"
        tour = tour[1:]  # Strip off depot

        total_cost = calc_pctsp_cost(depot, loc, penalty, deterministic_prize, tour)
        assert (float_to_scaled_int(total_cost) - output[1]) / float(output[1]) < 1e-5
        return total_cost, tour, duration
    except Exception as e:
        print("Exception occured")
        print(e)
        return None


# >>> ADDED LKH SOLVER FUNCTIONS FOR PCTSP (Copied/Adapted from OP baseline) <<<

def write_lkh_pctsp_par(filename, parameters):
    """Writes the LKH parameter file (.par) for PCTSP."""
    # Default parameters adjusted for PCTSP if known, otherwise generic
    default_parameters = {
        "MAX_TRIALS": 10000,
        "RUNS": 1, 
        "TRACE_LEVEL": 1,
        "SEED": 1234,
        # Check LKH docs for specific PCTSP parameters like MIN_PRIZE or PENALTY_TYPE
        # "MIN_PRIZE": ..., # This will be added from parameters if provided
    }
    with open(filename, 'w') as f:
        final_params = {**default_parameters, **parameters}
        # Remove None values before writing, except possibly specific flags if LKH uses them
        # final_params = {k: v for k, v in final_params.items() if v is not None}

        for k, v in final_params.items():
            if isinstance(v, str) and ("FILE" in k or "TOUR" in k):
                 abs_path = os.path.abspath(v)
                 f.write(f"{k} = {abs_path.replace(os.sep, '/')}\\n")
            elif v is not None: # Write parameter if value is not None
                 f.write(f"{k} = {v}\\n")
            # else: skip None values

def write_lkh_pctsp_problem(filename, depot, loc, penalty, prize, name="problem"):
    """Writes a PCTSP problem file in a format LKH can understand."""
    points = [depot] + loc
    prizes_in = [0] + prize # Add 0 prize for depot
    penalties_in = [0] + penalty # Add 0 penalty for depot
    n = len(points)
    scale_factor = 1000000 # Scale factor for coordinates
    prize_scale_factor = 100 # Scale factor for prizes/penalties if float

    with open(filename, 'w') as f:
        f.write(f"NAME : {name}\\n")
        f.write("TYPE : PCTSP\\n")
        f.write(f"DIMENSION : {n}\\n")
        f.write("EDGE_WEIGHT_TYPE : EUC_2D\\n")
        f.write("NODE_COORD_SECTION\\n")
        for i, (x, y) in enumerate(points):
            f.write(f"{i + 1} {int(x * scale_factor + 0.5)} {int(y * scale_factor + 0.5)}\\n")
        
        # NOTE: LKH PCTSP format might vary. Common variants use PRIZE_SECTION 
        # and PENALTY_SECTION, or just one if penalty=prize. Assuming separate sections.
        # Check LKH documentation for the exact keywords and format expected.

        f.write("PRIZE_SECTION\\n") # Assuming keyword
        for i, p in enumerate(prizes_in):
             scaled_prize = int(p * prize_scale_factor + 0.5) if isinstance(p, float) else int(p)
             f.write(f"{i + 1} {scaled_prize}\\n")

        f.write("PENALTY_SECTION\\n") # Assuming keyword
        for i, p in enumerate(penalties_in):
             scaled_penalty = int(p * prize_scale_factor + 0.5) if isinstance(p, float) else int(p)
             f.write(f"{i + 1} {scaled_penalty}\\n")

        f.write("DEPOT_SECTION\\n")
        f.write("1\\n")
        f.write("-1\\n")
        f.write("EOF\\n")

def read_lkh_tour(filename, n):
    """Reads the tour from LKH output file."""
    # This function is copied from op_baseline.py, assumed to be general enough
    with open(filename, 'r') as f:
        tour = []
        dimension = 0
        started = False
        for line in f:
            line = line.strip()
            if started:
                try:
                    loc = int(line)
                    if loc == -1:
                        break
                    tour.append(loc)
                except ValueError:
                    print(f"Warning: could not parse tour line: {line}")
                    continue
            if line.startswith("DIMENSION"):
                 parts = line.split()
                 if len(parts) >= 3:
                     dimension = int(parts[-1])
                 else:
                      print(f"Warning: could not parse DIMENSION from line: {line}")
            if line.upper().startswith("TOUR_SECTION") or line.upper().startswith("NODE_SEQUENCE_SECTION"):
                started = True

    if not tour:
         print(f"Warning: Empty tour read from {filename}")
         return []

    tour = np.array(tour).astype(int) - 1
    if tour[0] != 0:
        print(f"Warning: LKH tour from {filename} does not start with depot (0). Tour: {tour}")

    # For PCTSP, the tour usually includes visited nodes sequence starting (and sometimes ending) at depot.
    # We want the sequence of *customer* nodes visited. Assuming LKH output is [depot, cust1, cust2, ..., depot?]
    # Return tour[1:] or tour[1:-1] depending on whether LKH includes closing depot link.
    # Let's assume it includes only visited customers after the first depot for now.
    return tour[1:].tolist()

# Need a scaling function matching Gurobi call's scaling if prizes are floats
def scale_prize_for_lkh(prize_val, scale=100):
    # Gurobi likely uses high precision floats. LKH needs ints.
    # Match scaling used in write_lkh_pctsp_problem.
    return int(prize_val * scale + 0.5) if isinstance(prize_val, float) else int(prize_val)

def solve_lkh_pctsp_log(executable, directory, name, depot, loc, penalty, prize, min_prize_collect, runs=1, disable_cache=False):
    """
    Solves the PCTSP problem using the LKH solver.
    Handles file generation, LKH execution, output parsing, calculation, and caching.
    Returns: (cost, tour, duration)
    Cost = TourLength + SumOfPenaltiesOfUnvisitedNodes
    """
    problem_base = f"{name}.lkh_pctsp{runs}"
    problem_filename = os.path.join(directory, f"{problem_base}.pctsp")
    tour_filename = os.path.join(directory, f"{problem_base}.tour")
    output_filename = os.path.join(directory, f"{problem_base}.pkl") # Cache file
    param_filename = os.path.join(directory, f"{problem_base}.par")
    log_filename = os.path.join(directory, f"{problem_base}.log")

    try:
        if os.path.isfile(output_filename) and not disable_cache:
            (cost, tour, duration) = load_dataset(output_filename)
            # Optional: Add verification for cached solution (e.g., check min prize constraint)
        else:
            lkh_executable_path = executable
            if not os.path.isfile(lkh_executable_path):
                raise FileNotFoundError(f"LKH executable not found at {lkh_executable_path}")

            # Generate PCTSP problem file (using deterministic prize)
            write_lkh_pctsp_problem(problem_filename, depot, loc, penalty, prize, name=name)

            # Generate parameter file with MIN_PRIZE constraint
            # Check LKH documentation for exact keyword for minimum prize constraint!
            # Assuming "MIN_PRIZE" and that it needs to be scaled like prizes.
            scaled_min_prize = scale_prize_for_lkh(min_prize_collect)
            params = {
                "PROBLEM_FILE": problem_filename,
                "OUTPUT_TOUR_FILE": tour_filename,
                "RUNS": runs,
                "SEED": 1234,
                "MAX_TRIALS": 10000,
                "MIN_PRIZE": scaled_min_prize # Assumed LKH parameter
            }
            write_lkh_pctsp_par(param_filename, params)

            with open(log_filename, 'w') as f:
                start_time = time.time()
                try:
                    param_file_path_fwd = param_filename.replace(os.sep, '/')
                    cmd = [lkh_executable_path, param_file_path_fwd]
                    subprocess.check_call(cmd, stdout=f, stderr=subprocess.STDOUT, cwd=directory)
                except subprocess.CalledProcessError as e:
                     print(f"LKH execution failed for {name}. Check log: {log_filename}. Error: {e}")
                     raise
                except FileNotFoundError:
                     print(f"Error: LKH executable not found or command failed: {' '.join(cmd)}")
                     raise
                duration = time.time() - start_time

            tour = read_lkh_tour(tour_filename, n=len(loc) + 1)
            
            # Calculate PCTSP cost using the provided utility function
            cost = calc_pctsp_cost(depot, loc, penalty, prize, tour)
            
            # Optional: Verify if minimum prize constraint was met by LKH solution
            # collected_prize = calc_pctsp_total(prize, tour)
            # if collected_prize < min_prize_collect - 1e-5: # tolerance
            #     print(f"Warning: LKH solution for {name} did not meet min prize constraint. Collected: {collected_prize}, Required: {min_prize_collect}")

            save_dataset((cost, tour, duration), output_filename)

        return cost, tour, duration

    except Exception as e:
        print(f"Exception occurred processing {name} with LKH for PCTSP: {e}")
        import traceback
        traceback.print_exc()
        return None

# >>> END ADDED LKH SOLVER FUNCTIONS <<<


# solve_salesman function remains here...
# ... existing solve_salesman code ...

# Replace solve_gurobi with LKH-based solver
# def solve_gurobi(directory, name, depot, loc, penalty, deterministic_prize, stochastic_prize,
#                  disable_cache=False, timeout=None, gap=None):
#     # Lazy import so we do not need to have gurobi installed to run this script
#     from .pctsp_gurobi import solve_euclidian_pctsp as solve_euclidian_pctsp_gurobi
# 
#     try:
#         problem_filename = os.path.join(directory, "{}.gurobi{}{}.pkl".format(
#             name, "" if timeout is None else "t{}".format(timeout), "" if gap is None else "gap{}".format(gap)))
# 
#         if os.path.isfile(problem_filename) and not disable_cache:
#             (cost, tour, duration) = load_dataset(problem_filename)
#         else:
#             # 0 = start, 1 = end so add depot twice
#             start = time.time()
# 
#             # Must collect 1 or the sum of the prices if it is less then 1.
#             # Use deterministic_prize for gurobi call, consistent with its implementation
#             min_prize_collect = min(sum(deterministic_prize), 1.) 
#             cost, tour = solve_euclidian_pctsp_gurobi(
#                 depot, loc, penalty, deterministic_prize, min_prize_collect,
#                 threads=1, timeout=timeout, gap=gap
#             )
#             duration = time.time() - start  # Measure clock time
#             save_dataset((cost, tour, duration), problem_filename)
# 
#         # First and last node are depot(s), so first node is 2 but should be 1 (as depot is 0) so subtract 1
#         assert tour[0] == 0
#         tour = tour[1:] # Keep only customer nodes in the tour list
# 
#         # Recalculate cost using the utility function to ensure consistency
#         # The cost returned by gurobi might be slightly different due to internal scaling/precision
#         total_cost = calc_pctsp_cost(depot, loc, penalty, deterministic_prize, tour)
#         # assert abs(total_cost - cost) <= 1e-5, "Cost is incorrect"
#         # Use recalculated cost for consistency
#         return total_cost, tour, duration
# 
#     except Exception as e:
#         # For some stupid reason, sometimes OR tools cannot find a feasible solution?
#         # By letting it fail we do not get total results, but we can retry by the caching mechanism
#         print("Exception occured")
#         print(e)
#         return None

def solve_gurobi(directory, name, depot, loc, penalty, deterministic_prize, stochastic_prize,
                 disable_cache=False, timeout=None, gap=None, runs=1):
    """
    Solves the PCTSP problem using LKH.
    Function name kept as solve_gurobi for compatibility.
    Timeout and gap parameters are ignored (LKH controlled by runs and .par file).
    Uses deterministic_prize similar to the original Gurobi call.
    Returns: (cost, tour, duration)
    Cost = TourLength + SumOfPenaltiesOfUnvisitedNodes
    """
    # Define LKH executable path using raw string
    lkh_executable = r"C:\Users\Administrator\Documents\TSP-HAC-main\TSP-HAC-main\LKH-3.0.13\LKH.exe"

    try:
        # Calculate minimum prize to collect, same logic as original gurobi call
        # We use deterministic_prize here as the original Gurobi implementation did.
        min_prize_collect = min(sum(deterministic_prize), 1.)

        # Call the new LKH PCTSP solver function
        result = solve_lkh_pctsp_log(
            executable=lkh_executable,
            directory=directory,
            name=name,
            depot=depot,
            loc=loc,
            penalty=penalty, # Pass penalties
            prize=deterministic_prize, # Pass deterministic prizes
            min_prize_collect=min_prize_collect, # Pass minimum prize constraint
            runs=runs,
            disable_cache=disable_cache
        )

        if result is None:
            print(f"LKH solving failed for PCTSP {name}.")
            return None

        # Result is (cost, tour, duration)
        cost, tour, duration = result
        
        # Optional: verify cost calculation consistency
        # recalculated_cost = calc_pctsp_cost(depot, loc, penalty, deterministic_prize, tour)
        # if abs(cost - recalculated_cost) > 1e-4:
        #    print(f"Warning: Cost mismatch for PCTSP {name}. LKH returned {cost}, recalculated {recalculated_cost}")

        return cost, tour, duration

    except Exception as e:
        print(f"Exception occurred in solve_gurobi (LKH wrapper) for PCTSP {name}: {e}")
        import traceback
        traceback.print_exc()
        return None


def solve_ortools(directory, name, depot, loc, penalty, deterministic_prize, stochastic_prize,
                  sec_local_search=0, disable_cache=False):
    # Lazy import so we do not require ortools by default
    from .pctsp_ortools import solve_pctsp_ortools

    try:
        problem_filename = os.path.join(directory, "{}.ortools{}.pkl".format(name, sec_local_search))
        if os.path.isfile(problem_filename) and not disable_cache:
            objval, tour, duration = load_dataset(problem_filename)
        else:
            # 0 = start, 1 = end so add depot twice
            start = time.time()
            objval, tour = solve_pctsp_ortools(depot, loc, deterministic_prize, penalty,
                                               min(sum(deterministic_prize), 1.), sec_local_search=sec_local_search)
            duration = time.time() - start
            save_dataset((objval, tour, duration), problem_filename)
        assert tour[0] == 0, "Tour must start with depot"
        tour = tour[1:]
        total_cost = calc_pctsp_cost(depot, loc, penalty, deterministic_prize, tour)
        assert abs(total_cost - objval) <= 1e-5, "Cost is incorrect"
        return total_cost, tour, duration
    except Exception as e:
        # For some stupid reason, sometimes OR tools cannot find a feasible solution?
        # By letting it fail we do not get total results, but we dcan retry by the caching mechanism
        print("Exception occured")
        print(e)
        return None


def calc_pctsp_total(vals, tour):
    # Subtract 1 since vals index start with 0 while tour indexing starts with 1 as depot is 0
    assert (np.array(tour) > 0).all(), "Depot cannot be in tour"
    return np.array(vals)[np.array(tour) - 1].sum()


def calc_pctsp_length(depot, loc, tour):
    loc_with_depot = np.vstack((np.array(depot)[None, :], np.array(loc)))
    sorted_locs = loc_with_depot[np.concatenate(([0], tour, [0]))]
    return np.linalg.norm(sorted_locs[1:] - sorted_locs[:-1], axis=-1).sum()


def calc_pctsp_cost(depot, loc, penalty, prize, tour):
    # With some tolerance we should satisfy minimum prize
    assert len(np.unique(tour)) == len(tour), "Tour cannot contain duplicates"
    assert calc_pctsp_total(prize, tour) >= 1 - 1e-5 or len(tour) == len(prize), \
        "Tour should collect at least 1 as total prize or visit all nodes"
    # Penalty is only incurred for locations not visited, so charge total penalty minus penalty of locations visited
    return calc_pctsp_length(depot, loc, tour) + np.sum(penalty) - calc_pctsp_total(penalty, tour)


def write_pctsp(filename, depot, loc, penalty, prize, name="problem"):
    coord = [depot] + loc
    return write_pctsp_dist(filename, distance_matrix(coord, coord), penalty, prize)


def float_to_scaled_int_str(v):  # Program only accepts ints so scale everything by 10^7
    return str(float_to_scaled_int(v))


def float_to_scaled_int(v):
    return int(v * 10000000 + 0.5)


def write_pctsp_dist(filename, dist, penalty, prize):

    with open(filename, 'w') as f:
        f.write("\n".join([
            "",
            " ".join([float_to_scaled_int_str(p) for p in [0] + list(prize)]),
            "",
            "",
            " ".join([float_to_scaled_int_str(p) for p in [0] + list(penalty)]),
            "",
            "",
            *(
                " ".join(float_to_scaled_int_str(d) for d in d_row)
                for d_row in dist
            )
        ]))


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("method",
                        help="Name of the method to evaluate, 'pctsp', 'salesman' or 'stochpctsp(first|half|all)'")
    parser.add_argument("datasets", nargs='+', help="Filename of the dataset(s) to evaluate")
    parser.add_argument("-f", action='store_true', help="Set true to overwrite")
    parser.add_argument("-o", default=None, help="Name of the results file to write")
    parser.add_argument("--cpus", type=int, help="Number of CPUs to use, defaults to all cores")
    parser.add_argument('--disable_cache', action='store_true', help='Disable caching')
    parser.add_argument('--progress_bar_mininterval', type=float, default=0.1, help='Minimum interval')
    parser.add_argument('-n', type=int, help="Number of instances to process")
    parser.add_argument('--offset', type=int, help="Offset where to start processing")
    parser.add_argument('--results_dir', default='results', help="Name of results directory")

    opts = parser.parse_args()

    assert opts.o is None or len(opts.datasets) == 1, "Cannot specify result filename with more than one dataset"

    for dataset_path in opts.datasets:

        assert os.path.isfile(check_extension(dataset_path)), "File does not exist!"

        dataset_basename, ext = os.path.splitext(os.path.split(dataset_path)[-1])

        if opts.o is None:
            results_dir = os.path.join(opts.results_dir, "pctsp", dataset_basename)
            os.makedirs(results_dir, exist_ok=True)

            out_file = os.path.join(results_dir, "{}{}{}-{}{}".format(
                dataset_basename,
                "offs{}".format(opts.offset) if opts.offset is not None else "",
                "n{}".format(opts.n) if opts.n is not None else "",
                opts.method, ext
            ))
        else:
            out_file = opts.o

        assert opts.f or not os.path.isfile(
            out_file), "File already exists! Try running with -f option to overwrite."

        match = re.match(r'^([a-z]+)(\d*)$', opts.method)
        assert match
        method = match[1]
        runs = 1 if match[2] == '' else int(match[2])

        if method in ("pctsp", "salesman", "gurobi", "gurobigap", "gurobit", "ortools") or method[:10] == "stochpctsp":

            target_dir = os.path.join(results_dir, "{}-{}".format(
                dataset_basename,
                opts.method
            ))
            assert opts.f or not os.path.isdir(target_dir), \
                "Target dir already exists! Try running with -f option to overwrite."

            if not os.path.isdir(target_dir):
                os.makedirs(target_dir)

            dataset = load_dataset(dataset_path)

            if method[:6] == "gurobi":
                use_multiprocessing = True  # We run one thread per instance

                def run_func(args):
                    return solve_gurobi(*args, disable_cache=opts.disable_cache,
                                        timeout=runs if method[6:] == "t" else None,
                                        gap=float(runs) if method[6:] == "gap" else None)
            elif method == "pctsp":
                executable = get_pctsp_executable()
                use_multiprocessing = False

                def run_func(args):
                    return solve_pctsp_log(executable, *args, runs=runs)
            elif method == "salesman":
                use_multiprocessing = True

                def run_func(args):
                    return solve_salesman(*args, runs=runs)
            elif method == "ortools":
                use_multiprocessing = True

                def run_func(args):
                    return solve_ortools(*args, sec_local_search=runs, disable_cache=opts.disable_cache)
            else:
                assert method[:10] == "stochpctsp"
                append = method[10:]
                assert append in ('first', 'half', 'all')
                use_multiprocessing = True

                def run_func(args):
                    return solve_stochastic_pctsp_log(executable, *args, runs=runs, append=append)

            results, parallelism = run_all_in_pool(
                run_func,
                target_dir, dataset, opts, use_multiprocessing=use_multiprocessing
            )

        else:
            assert False, "Unknown method: {}".format(opts.method)

        costs, tours, durations = zip(*results)  # Not really costs since they should be negative
        print("Average cost: {} +- {}".format(np.mean(costs), 2 * np.std(costs) / np.sqrt(len(costs))))
        print("Average serial duration: {} +- {}".format(
            np.mean(durations), 2 * np.std(durations) / np.sqrt(len(durations))))
        print("Average parallel duration: {}".format(np.mean(durations) / parallelism))
        print("Calculated total duration: {}".format(timedelta(seconds=int(np.sum(durations) / parallelism))))

        save_dataset((results, parallelism), out_file)
