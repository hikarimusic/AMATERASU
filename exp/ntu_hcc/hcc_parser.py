import csv

def parse_sra_data(filename):
    samples = []
    current_sample = {}
    
    with open(filename, 'r') as file:
        for line in file:
            if line.startswith('^SAMPLE ='):
                if current_sample:
                    samples.append(current_sample)
                current_sample = {'sample_id': line.split('=')[1].strip()}
            elif line.startswith('!Sample_characteristics_ch1'):
                key, value = line.split(':', 1)
                key = key.split('=')[1].strip()
                value = value.strip()
                if key in ['etiology', 'ctnnb1 status', 'efs days', 'efs event', 'os days', 'os event']:
                    current_sample[key] = value
    
    if current_sample:
        samples.append(current_sample)
    
    return samples

def write_csv(samples, output_file):
    fieldnames = ['sample_id', 'etiology', 'ctnnb1 status', 'efs days', 'efs event', 'os days', 'os event']
    
    with open(output_file, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for sample in samples:
            writer.writerow(sample)

# Main execution
input_file = "GSE141198_family.soft"
output_file = "hcc_family.csv"

samples = parse_sra_data(input_file)
write_csv(samples, output_file)

print(f"CSV file '{output_file}' has been created with {len(samples)} samples.")