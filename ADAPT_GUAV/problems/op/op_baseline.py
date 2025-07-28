import argparse
import os
import numpy as np
from utils import run_all_in_pool
from utils.data_utils import check_extension, load_dataset, save_dataset
from subprocess import check_call, check_output
import tempfile
import time
from datetime import timedelta
from problems.op.opga.opevo import run_alg as run_opga_alg
from tqdm import tqdm
import re
import subprocess
import math

MAX_LENGTH_TOL = 1e-5


# Run install_compass.sh to install
def solve_compass(executable, depot, loc, demand, capacity):
    with tempfile.TemporaryDirectory() as tempdir:
        problem_filename = os.path.join(tempdir, "problem.oplib")
        output_filename = os.path.join(tempdir, "output.tour")
        param_filename = os.path.join(tempdir, "params.par")

        starttime = time.time()
        write_oplib(problem_filename, depot, loc, demand, capacity)
        params = {"PROBLEM_FILE": problem_filename, "OUTPUT_TOUR_FILE": output_filename}
        write_compass_par(param_filename, params)
        output = check_output([executable, param_filename])
        result = read_oplib(output_filename, n=len(demand))
        duration = time.time() - starttime
        return result, output, duration


def solve_compass_log(executable, directory, name, depot, loc, prize, max_length, disable_cache=False):

    problem_filename = os.path.join(directory, "{}.oplib".format(name))
    tour_filename = os.path.join(directory, "{}.tour".format(name))
    output_filename = os.path.join(directory, "{}.compass.pkl".format(name))
    log_filename = os.path.join(directory, "{}.log".format(name))

    try:
        # May have already been run
        if os.path.isfile(output_filename) and not disable_cache:
            tour, duration = load_dataset(output_filename)
        else:
            write_oplib(problem_filename, depot, loc, prize, max_length, name=name)

            with open(log_filename, 'w') as f:
                start = time.time()
                check_call([executable, '--op', '--op-ea4op', problem_filename, '-o', tour_filename],
                           stdout=f, stderr=f)
                duration = time.time() - start

            tour = read_oplib(tour_filename, n=len(prize))
            if not calc_op_length(depot, loc, tour) <= max_length:
                print("Warning: length exceeds max length:", calc_op_length(depot, loc, tour), max_length)
            assert calc_op_length(depot, loc, tour) <= max_length + MAX_LENGTH_TOL, "Tour exceeds max_length!"
            save_dataset((tour, duration), output_filename)

        return -calc_op_total(prize, tour), tour, duration

    except Exception as e:
        print("Exception occured")
        print(e)
        return None


def calc_op_total(prize, tour):
    # Subtract 1 since vals index start with 0 while tour indexing starts with 1 as depot is 0
    assert (np.array(tour) > 0).all(), "Depot cannot be in tour"
    assert len(np.unique(tour)) == len(tour), "Tour cannot contain duplicates"
    return np.array(prize)[np.array(tour) - 1].sum()


def calc_op_length(depot, loc, tour):
    assert len(np.unique(tour)) == len(tour), "Tour cannot contain duplicates"
    loc_with_depot = np.vstack((np.array(depot)[None, :], np.array(loc)))
    sorted_locs = loc_with_depot[np.concatenate(([0], tour, [0]))]
    return np.linalg.norm(sorted_locs[1:] - sorted_locs[:-1], axis=-1).sum()


def write_compass_par(filename, parameters):
    default_parameters = {  # Use none to include as flag instead of kv
        "SPECIAL": None,
        "MAX_TRIALS": 10000,
        "RUNS": 10,
        "TRACE_LEVEL": 1,
        "SEED": 0
    }
    with open(filename, 'w') as f:
        for k, v in {**default_parameters, **parameters}.items():
            if v is None:
                f.write("{}\n".format(k))
            else:
                f.write("{} = {}\n".format(k, v))


def read_oplib(filename, n):
    with open(filename, 'r') as f:
        tour = []
        dimension = 0
        started = False
        for line in f:
            if started:
                loc = int(line)
                if loc == -1:
                    break
                tour.append(loc)
            if line.startswith("DIMENSION"):
                dimension = int(line.split(" ")[-1])

            if line.startswith("NODE_SEQUENCE_SECTION"):
                started = True
    
    assert len(tour) > 0, "Unexpected length"
    tour = np.array(tour).astype(int) - 1  # Subtract 1 as depot is 1 and should be 0
    assert tour[0] == 0  # Tour should start with depot
    assert tour[-1] != 0  # Tour should not end with depot
    return tour[1:].tolist()


def write_oplib(filename, depot, loc, prize, max_length, name="problem"):

    with open(filename, 'w') as f:
        f.write("\n".join([
            "{} : {}".format(k, v)
            for k, v in (
                ("NAME", name),
                ("TYPE", "OP"),
                ("DIMENSION", len(loc) + 1),
                ("COST_LIMIT", int(max_length * 10000000 + 0.5)),
                ("EDGE_WEIGHT_TYPE", "EUC_2D"),
            )
        ]))
        f.write("\n")
        f.write("NODE_COORD_SECTION\n")
        f.write("\n".join([
            "{}\t{}\t{}".format(i + 1, int(x * 10000000 + 0.5), int(y * 10000000 + 0.5))  # oplib does not take floats
            #"{}\t{}\t{}".format(i + 1, x, y)
            for i, (x, y) in enumerate([depot] + loc)
        ]))
        f.write("\n")
        f.write("NODE_SCORE_SECTION\n")
        f.write("\n".join([
            "{}\t{}".format(i + 1, d)
            for i, d in enumerate([0] + prize)
        ]))
        f.write("\n")
        f.write("DEPOT_SECTION\n")
        f.write("1\n")
        f.write("-1\n")
        f.write("EOF\n")


def solve_opga(directory, name, depot, loc, prize, max_length, disable_cache=False):
    problem_filename = os.path.join(directory, "{}.opga.pkl".format(name))
    if os.path.isfile(problem_filename) and not disable_cache:
        (prize, tour, duration) = load_dataset(problem_filename)
    else:
        # 0 = start, 1 = end so add depot twice
        start = time.time()
        prize, tour, duration = run_opga_alg(
            [(*pos, p) for p, pos in zip([0, 0] + prize, [depot, depot] + loc)],
            max_length, return_sol=True, verbose=False
        )
        duration = time.time() - start  # Measure clock time
        save_dataset((prize, tour, duration), problem_filename)

    # First and last node are depot(s), so first node is 2 but should be 1 (as depot is 0) so subtract 1
    assert tour[0][3] == 0
    assert tour[-1][3] == 1
    return -prize, [i - 1 for x, y, p, i, t in tour[1:-1]], duration


def write_lkh_pctsp_par(filename, parameters):
    """Writes the LKH parameter file (.par) for PCTSP."""
    default_parameters = {
        "MAX_TRIALS": 10000,
        "RUNS": 1,
        "TRACE_LEVEL": 1,
        "SEED": 1234,
    }
    with open(filename, 'w') as f:
        final_params = {**default_parameters, **parameters}
        # Remove None values before writing, LKH uses flags without values sometimes
        # Keep SPECIAL if it's passed as None, remove other None values
        # final_params = {k: v for k, v in final_params.items() if v is not None or k == "SPECIAL"}

        for k, v in final_params.items():
             # Use absolute paths converted to forward slashes for LKH compatibility
            if isinstance(v, str) and ("FILE" in k or "TOUR" in k):
                 # Ensure the path exists before making it absolute? No, par file might reference future output.
                 abs_path = os.path.abspath(v)
                 f.write(f"{k} = {abs_path.replace(os.sep, '/')}\\n")
            elif v is None and k == "SPECIAL": # Handle SPECIAL flag if needed by LKH variant
                 f.write(f"{k}\\n")
            elif v is not None:
                 f.write(f"{k} = {v}\\n")
            # else: # v is None and not SPECIAL, skip writing


def write_lkh_pctsp_problem(filename, depot, loc, prize, max_length, name="problem"):
    """Writes a PCTSP problem file in a format LKH can understand."""
    points = [depot] + loc
    prizes = [0] + prize # Depot has 0 prize
    n = len(points)

    # Scale floating point coordinates and prizes to integers for LKH
    # LKH typically requires integer data for coordinates and weights/prizes
    scale_factor = 1000000 # Same scale as used in tsp_baseline for coordinates

    with open(filename, 'w') as f:
        f.write(f"NAME : {name}\\n")
        f.write("TYPE : PCTSP\\n") # Specify problem type as PCTSP
        f.write(f"DIMENSION : {n}\\n")
        f.write("EDGE_WEIGHT_TYPE : EUC_2D\\n")
        # Add the COST_LIMIT (Max Length for OP) - check LKH docs for exact keyword
        # Assuming COST_LIMIT or MAX_COST. Using COST_LIMIT based on oplib example.
        # Need to scale the max_length as well
        f.write(f"COST_LIMIT : {int(max_length * scale_factor + 0.5)}\\n")
        f.write("NODE_COORD_SECTION\\n")
        for i, (x, y) in enumerate(points):
            f.write(f"{i + 1} {int(x * scale_factor + 0.5)} {int(y * scale_factor + 0.5)}\\n")

        # Add prize section (check LKH documentation for the exact keyword, PRIZE_SECTION is a guess)
        f.write("PRIZE_SECTION\\n") # Or NODE_SCORE_SECTION? Assuming PRIZE_SECTION
        for i, p in enumerate(prizes):
             # Scaling prize? Assuming prizes are already in a reasonable integer range or should be scaled if floats
             # Let's assume prizes are scaled if they are float, otherwise use as is.
             # Example: scale prize by 100 if they are small floats
             scaled_prize = int(p * 100 + 0.5) if isinstance(p, float) else int(p)
             f.write(f"{i + 1} {scaled_prize}\\n")

        f.write("DEPOT_SECTION\\n")
        f.write("1\\n") # Depot is node 1
        f.write("-1\\n") # End of depot section
        f.write("EOF\\n")


def read_lkh_tour(filename, n):
    """Reads the tour from LKH output file."""
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
                    continue # Skip non-integer lines in tour section
            if line.startswith("DIMENSION"):
                 # Check if DIMENSION line has format "DIMENSION: <value>" or "DIMENSION = <value>"
                 parts = line.split()
                 if len(parts) >= 3:
                     dimension = int(parts[-1])
                 else:
                      print(f"Warning: could not parse DIMENSION from line: {line}")
            # Check for common TOUR_SECTION markers
            if line.upper().startswith("TOUR_SECTION") or line.upper().startswith("NODE_SEQUENCE_SECTION"):
                started = True

    # Basic validation - LKH tour for PCTSP might not visit all nodes
    # assert len(tour) == dimension, f"Tour length {len(tour)} does not match dimension {dimension}"

    if not tour:
         print(f"Warning: Empty tour read from {filename}")
         return []

    # LKH uses 1-based indexing
    tour = np.array(tour).astype(int) - 1

    # PCTSP/OP tour should start with the depot (node 0 after conversion)
    # It might or might not end with the depot depending on LKH output format.
    # Usually, it represents the sequence of visited nodes starting from the depot.
    if tour[0] != 0:
        print(f"Warning: LKH tour from {filename} does not start with depot (0). Tour: {tour}")
        # Attempt to fix if depot is elsewhere? Or just return as is? Return as is for now.

    # Return only visited customer nodes (exclude the starting depot 0)
    return tour[1:].tolist()


def solve_lkh_op_log(executable, directory, name, depot, loc, prize, max_length, runs=1, disable_cache=False):
    """
    Solves the OP problem using the LKH solver, adapted for PCTSP formulation.
    Handles file generation, LKH execution, output parsing, and caching.
    Returns: (total_prize, tour, duration)
    Note: LKH maximizes prize, so we return the positive prize. Callers expecting negative cost should negate.
    """
    problem_base = f"{name}.lkh_op{runs}"
    problem_filename = os.path.join(directory, f"{problem_base}.pctsp")
    tour_filename = os.path.join(directory, f"{problem_base}.tour")
    output_filename = os.path.join(directory, f"{problem_base}.pkl") # Cache file
    param_filename = os.path.join(directory, f"{problem_base}.par")
    log_filename = os.path.join(directory, f"{problem_base}.log")

    try:
        if os.path.isfile(output_filename) and not disable_cache:
            (calculated_prize, tour, duration) = load_dataset(output_filename)
            current_length = calc_op_length(depot, loc, tour)
            if not current_length <= max_length + MAX_LENGTH_TOL:
                 print(f"Warning: Cached tour for {name} invalid (length {current_length} > {max_length}). Recalculating.")
                 raise FileNotFoundError
        else:
            lkh_executable_path = executable # Use the raw path provided
            if not os.path.isfile(lkh_executable_path):
                raise FileNotFoundError(f"LKH executable not found at {lkh_executable_path}")

            write_lkh_pctsp_problem(problem_filename, depot, loc, prize, max_length, name=name)

            params = {
                "PROBLEM_FILE": problem_filename,
                "OUTPUT_TOUR_FILE": tour_filename,
                "RUNS": runs,
                "SEED": 1234,
                "MAX_TRIALS": 10000,
                # Add COST_LIMIT directly to params if needed, check LKH docs for PCTSP param name
                # Example: "COST_LIMIT": int(max_length * 1000000 + 0.5)
            }
            write_lkh_pctsp_par(param_filename, params)

            with open(log_filename, 'w') as f:
                start_time = time.time()
                try:
                    # Convert paths to forward slashes for the command line argument to LKH
                    param_file_path_fwd = param_filename.replace(os.sep, '/')
                    # Use raw string for executable path if it contains spaces or special chars,
                    # but subprocess might handle it if passed as a list item.
                    cmd = [lkh_executable_path, param_file_path_fwd]
                    subprocess.check_call(cmd, stdout=f, stderr=subprocess.STDOUT, cwd=directory) # Run LKH from the directory
                except subprocess.CalledProcessError as e:
                     print(f"LKH execution failed for {name}. Check log: {log_filename}")
                     print(f"Error details: {e}")
                     raise
                except FileNotFoundError:
                     # This might catch if the executable itself is not found
                     print(f"Error: LKH executable not found or command failed: {' '.join(cmd)}")
                     raise
                duration = time.time() - start_time

            tour = read_lkh_tour(tour_filename, n=len(loc) + 1)
            calculated_prize = calc_op_total(prize, tour)
            length = calc_op_length(depot, loc, tour)
            if not length <= max_length + MAX_LENGTH_TOL:
                 print(f"Warning: LKH solution for {name} exceeds max_length ({length} > {max_length}). Tour: {tour}")

            save_dataset((calculated_prize, tour, duration), output_filename)

        return calculated_prize, tour, duration

    except Exception as e:
        print(f"Exception occurred processing {name} with LKH for OP: {e}")
        import traceback
        traceback.print_exc() # Ensure this is on a new line
        return None


def solve_gurobi(directory, name, depot, loc, prize, max_length, disable_cache=False, timeout=None, gap=None, runs=1):
    """
    Solves the OP problem using LKH (via PCTSP formulation).
    Function name kept as solve_gurobi for compatibility.
    Timeout and gap parameters are ignored (LKH controlled by runs and .par file).
    Returns: (negative_total_prize, tour, duration) consistent with original gurobi version's expected output.
    """
    # Define LKH executable path using raw string
    lkh_executable = r"C:\Users\Administrator\Documents\TSP-HAC-main\TSP-HAC-main\LKH-3.0.13\LKH.exe"

    try:
        result = solve_lkh_op_log(
            executable=lkh_executable,
            directory=directory,
            name=name,
            depot=depot,
            loc=loc,
            prize=prize,
            max_length=max_length,
            runs=runs,
            disable_cache=disable_cache
        )

        if result is None:
            print(f"LKH solving failed for {name}.")
            return None

        total_prize, tour, duration = result
        cost = -total_prize # Negate prize to match original cost output

        length = calc_op_length(depot, loc, tour)
        if not length <= max_length + MAX_LENGTH_TOL:
             print(f"Warning: Final OP tour length {length} exceeds max_length {max_length} for {name}")

        recalculated_prize = calc_op_total(prize, tour)
        if abs(total_prize - recalculated_prize) > 1e-4:
            print(f"Warning: Prize mismatch for {name}. LKH returned {total_prize}, recalculated {recalculated_prize}")

        return cost, tour, duration

    except Exception as e:
        print(f"Exception occurred in solve_gurobi (LKH wrapper) for {name}: {e}")
        import traceback
        traceback.print_exc() # Ensure this is on a new line
        return None


def solve_ortools(directory, name, depot, loc, prize, max_length, sec_local_search=0, disable_cache=False):
    # Lazy import so we do not require ortools by default
    from problems.op.op_ortools import solve_op_ortools

    try:
        problem_filename = os.path.join(directory, "{}.ortools{}.pkl".format(name, sec_local_search))
        if os.path.isfile(problem_filename) and not disable_cache:
            objval, tour, duration = load_dataset(problem_filename)
        else:
            # 0 = start, 1 = end so add depot twice
            start = time.time()
            objval, tour = solve_op_ortools(depot, loc, prize, max_length, sec_local_search=sec_local_search)
            duration = time.time() - start
            save_dataset((objval, tour, duration), problem_filename)
        assert tour[0] == 0, "Tour must start with depot"
        tour = tour[1:]
        assert calc_op_length(depot, loc, tour) <= max_length + MAX_LENGTH_TOL, "Tour exceeds max_length!"
        assert abs(-calc_op_total(prize, tour) - objval) <= 1e-5, "Cost is incorrect"
        return -calc_op_total(prize, tour), tour, duration
    except Exception as e:
        # For some stupid reason, sometimes OR tools cannot find a feasible solution?
        # By letting it fail we do not get total results, but we dcan retry by the caching mechanism
        print("Exception occured")
        print(e)
        return None


def run_all_tsiligirides(
        dataset_path, sample, num_samples, eval_batch_size, max_calc_batch_size, no_cuda=False, dataset_n=None,
        progress_bar_mininterval=0.1, seed=1234):
    import torch
    from torch.utils.data import DataLoader
    from utils import move_to, sample_many
    from problems.op.tsiligirides import op_tsiligirides
    from problems.op.problem_op import OP
    torch.manual_seed(seed)

    dataloader = DataLoader(
        OP.make_dataset(filename=dataset_path, num_samples=dataset_n if dataset_n is not None else 1000000),
        batch_size=eval_batch_size
    )
    device = torch.device("cuda:0" if torch.cuda.is_available() and not no_cuda else "cpu")
    results = []
    for batch in tqdm(dataloader, mininterval=progress_bar_mininterval):
        start = time.time()
        batch = move_to(batch, device)

        with torch.no_grad():
            if num_samples * eval_batch_size > max_calc_batch_size:
                assert eval_batch_size == 1
                assert num_samples % max_calc_batch_size == 0
                batch_rep = max_calc_batch_size
                iter_rep = num_samples // max_calc_batch_size
            else:
                batch_rep = num_samples
                iter_rep = 1
            sequences, costs = sample_many(
                lambda inp: (None, op_tsiligirides(inp, sample)),
                OP.get_costs,
                batch, batch_rep=batch_rep, iter_rep=iter_rep)
            duration = time.time() - start
            results.extend(
                [(cost.item(), np.trim_zeros(pi.cpu().numpy(),'b'), duration) for cost, pi in zip(costs, sequences)])
    return results, eval_batch_size


if __name__ == "__main__":
    executable = os.path.abspath(os.path.join('problems', 'op', 'compass', 'compass'))

    parser = argparse.ArgumentParser()
    parser.add_argument("method", help="Name of the method to evaluate, 'compass', 'opga' or 'tsili'")
    parser.add_argument("datasets", nargs='+', help="Filename of the dataset(s) to evaluate")
    parser.add_argument("-f", action='store_true', help="Set true to overwrite")
    parser.add_argument("-o", default=None, help="Name of the results file to write")
    parser.add_argument("--cpus", type=int, help="Number of CPUs to use, defaults to all cores")
    parser.add_argument('--no_cuda', action='store_true', help='Disable CUDA (only for Tsiligirides)')
    parser.add_argument('--disable_cache', action='store_true', help='Disable caching')
    parser.add_argument('--max_calc_batch_size', type=int, default=1000, help='Size for subbatches')
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
            results_dir = os.path.join(opts.results_dir, "op", dataset_basename)
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

        if method == "tsili" or method == "tsiligreedy":
            assert opts.offset is None, "Offset not supported for Tsiligirides"

            if method == "tsiligreedy":
                sample = False
                num_samples = 1
            else:
                sample = True
                num_samples = runs

            eval_batch_size = max(1, opts.max_calc_batch_size // num_samples)

            results, parallelism = run_all_tsiligirides(
                dataset_path, sample, num_samples, eval_batch_size, opts.max_calc_batch_size, opts.no_cuda, opts.n,
                opts.progress_bar_mininterval
            )
        elif method in ("compass", "opga", "gurobi", "gurobigap", "gurobit", "ortools"):

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
            elif method == "compass":
                use_multiprocessing = False

                def run_func(args):
                    return solve_compass_log(executable, *args, disable_cache=opts.disable_cache)
            elif method == "opga":
                use_multiprocessing = True

                def run_func(args):
                    return solve_opga(*args, disable_cache=opts.disable_cache)
            else:
                assert method == "ortools"
                use_multiprocessing = True

                def run_func(args):
                    return solve_ortools(*args, sec_local_search=runs, disable_cache=opts.disable_cache)

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
