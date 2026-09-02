import re
import numpy as np


def parse_chemkin_file(filename):
    species_data = {}
    with open(filename, 'r') as file:
        lines = file.readlines()

    i = 0
    while i < len(lines):
        if lines[i].strip():
            species_line = lines[i].strip()
            species_name = species_line[:24].strip()
            temp_ranges = list(map(float, species_line[45:].split()))
            coeffs = []
            for j in range(1, 4):
                line = lines[i + j].strip()
                # Use regex to find all 16-character floats in the line
                matches = re.findall(r'[+-]?\d+\.\d+[Ee][+-]?\d+', line)
                coeffs.extend(list(map(float, matches)))
            species_data[species_name] = {
                'coeffs': coeffs,
                'temp_ranges': temp_ranges
            }
            i += 4
        else:
            i += 1
    return species_data


def calculate_properties(coeffs, T):
    t = T / 1000.0
    cp = (coeffs[0] + coeffs[1] * t + coeffs[2] * t ** 2 + coeffs[3] * t ** 3 + coeffs[
        4] * t ** 4) * 0.239005736  # J/mol*K to cal/mol*K
    h = (coeffs[0] * t + coeffs[1] * t ** 2 / 2 + coeffs[2] * t ** 3 / 3 + coeffs[3] * t ** 4 / 4 + coeffs[
        4] * t ** 5 / 5 + coeffs[5] / t) * 2.39005736e-4  # J/mol to kcal/mol
    s = (coeffs[0] * np.log(t) + coeffs[1] * t + coeffs[2] * t ** 2 / 2 + coeffs[3] * t ** 3 / 3 + coeffs[
        4] * t ** 4 / 4 + coeffs[6]) * 0.239005736  # J/mol*K to cal/mol*K
    return cp, h, s


def main():
    input_file = 'local_dataset'
    output_file = 'output_dataset'
    temperatures = [298.15, 300, 400, 500, 600, 800, 1000, 1500]

    species_data = parse_chemkin_file(input_file)

    with open(output_file, 'w') as f:
        for species, data in species_data.items():
            coeffs = data['coeffs']
            f.write(f'Species: {species}\n')
            f.write(f'Temperature(K)  Cp(cal/mol*K)   H(kcal/mol)   S(cal/mol*K)\n')
            for T in temperatures:
                cp, h, s = calculate_properties(coeffs, T)
                f.write(f'{T:<15} {cp:<15.4f} {h:<15.4f} {s:<15.4f}\n')
            f.write('\n')


if __name__ == '__main__':
    main()