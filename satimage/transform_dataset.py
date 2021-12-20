import argparse
from itertools import product
import sys

def main(input_file):

    id_to_label_mapping = {
        '1': "red soil",
        '2': "cotton crop",
        '3': "grey soil",
        '4': "damp grey soil",
        '5': "soil with vegetation stubble",
        '6': "mixture class (all types present)",
        '7': "very damp grey soil",
    }

    file = open(input_file)
    out = open(f'{input_file}.csv', 'w')
    pixels = ["p" + str(i + 1) for i in range(9)]
    bands = ["Green" , "Red", "NIR", "NIR"]
    names = list(product(pixels, bands))
    names = list(map(lambda x: x[0] + "." + x[1], names))
    names.append("Label")
    names = ",".join(names)
    out.write(names + "\n")
    for line in file:
        values = line.split()
        values[-1] = id_to_label_mapping[values[-1]].title()
        out.write(",".join(values) + "\n")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Transform the ImageSat dataset into csv format')
    parser.add_argument('file', type=str, metavar='INPUT_FILE', nargs=1)
    args = parser.parse_args()
    main(args.file[0])
    
