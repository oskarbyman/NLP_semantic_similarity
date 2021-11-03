import argparse
import csv
import wnhier
import wnNounSim
import wnSemSim

def handle_csv(file):
    """
    Handles csv files by parsing them into a list.
    Requires the delimiter to be a semi-colon.

    Params: 
        file: a csv filename
    Returns:
        a list with the csv file contents
    """
    with open(file, newline='', encoding="utf8") as f:
        reader = csv.reader(f, delimiter=';')
        data = list(reader)
    return data

def run_script(mode, data, output_file=None):
    """
    Runs either the hierarchical reasoning or the Wu-Palmer similarity script

    Params:
        mode: Either hierarchical or wu-palmer
        data: a two dimensional list, 
            where sentence pairs are the first and second element of a row
        output_file: filename where the results will be appended
    Returns:
        result_list: list of results if no output file is specified.
    """
    results = []
    for i in range(0, len(data)):
        S1 = data[i][0]
        S2 = data[i][1]
        if mode == "hierarchical":
            results.append(wnhier.Sim(S1, S2))
        elif mode == "wu-palmer":
            results.append(wnNounSim.Similarity(S1, S2))
        elif mode == "idf":
            results.append(wnSemSim.Similarity(S1, S2))
    if output_file:
        with open(output_file, 'a', newline='') as f:
            writer = csv.writer(f, delimiter=";")
            for i, row in enumerate(results):
                writer.writerow([data[i][0], data[i][1], results[i]])
    else:
        result_list = []
        print("Results:")
        for i, row in enumerate(results):
            result_list.append([data[i][0], data[i][1], results[i]])
            print(f"Sentence 1: {data[i][0]}, Sentence 2: {data[i][1]}, Similarity: {results[i]}")
        return result_list

def main():
    """
    A Command Line Interface for running either the hierarchical or wu-palmer similarity script
    Help with usage: python main.py --help
    """
    parser = argparse.ArgumentParser(description="Run semantic similarity scripts on csv:s, text files or sentences")
    subparsers = parser.add_subparsers(help="Sub-programs: ")
    parser_test = subparsers.add_parser("test", help="For testing the programs scripts against the STSS-131 dataset. Use 'test -h', for help.")
    parser_test.add_argument("test", help="Run a test of the chosen similarity script", default=True)

    parser_use = subparsers.add_parser("use", help="For actual usage of the program. Use 'use -h', for help.")
    input_group = parser_use.add_mutually_exclusive_group(required=True)
    input_group.add_argument("-c", "--csv", help="Input csv file, must have sentences as the first and second element of each row")
    input_group.add_argument("-s", "--sentences", help="semi-colon separated sentences, e.g. 'The weather is sunny';'The moon is not present'")
    parser_use.add_argument("-m", "--mode", help="Choose the mode of similarity measurement", choices=["hierarchical", "wu-palmer", "idf"], required=True)
    parser_use.add_argument("-o", "--output", help="Name of the output file, if empty results will be printed and returned")
    args = parser.parse_args()

    if hasattr(args, "test"):
        print("Running The Inverse Document Frequency based semantic similarity script...")
        wnSemSim.main()
        print("Running Wu-Palmer Noun derivation based semantic similarity script...")
        wnNounSim.main()
        print("Running Hierarchical reasoning semantic similarity test script...")
        wnhier.test_script()
    else:
        if args.csv:
            data = handle_csv(args.csv)
        elif args.sentences:
            data = [args.sentences.split(";")]
        else:
            raise RuntimeError("No input data given")
        result = run_script(args.mode, data, args.output)
        if result:
            return result
        else:
            print(f"Results saved to {args.output}")

    
if __name__ == "__main__":
    main()