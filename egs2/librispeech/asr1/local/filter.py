#!/usr/bin/env python3
import re
from decimal import Decimal

def is_time_event_with_decimal(line):
    """Check if line matches T{num} where num has a decimal point."""
    line = line.strip()
    # Match T followed by a number with a decimal point
    pattern = r'^T\d+\.\d+$'
    return bool(re.match(pattern, line))

def format_number(num):
    """Format number, truncating trailing zeros in hundredths place."""
    # Convert to string with 2 decimal places
    s = f"{num:.2f}"
    return s

def process_file(input_file, output_file):
    """Process the text file according to specifications."""
    
    print("Reading and filtering lines...")
    # Step 1: Read file and filter out T{decimal} lines
    filtered_lines = []
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            stripped = line.strip()
            if not is_time_event_with_decimal(stripped):
                filtered_lines.append(stripped)
    
    print(f"Filtered: {len(filtered_lines)} lines remaining")
    
    # Step 2: Sort the remaining lines
    print("Sorting lines...")
    filtered_lines.sort()
    
    # Step 3: Generate T{num} lines from 0.00 to 30.00 with 0.01 increment
    print("Generating T{num} lines...")
    time_lines = []
    
    # Use Decimal for precise floating point arithmetic
    current = Decimal('0.00')
    end = Decimal('30.00')
    increment = Decimal('0.01')
    
    while current <= end:
        num = float(current)
        formatted = format_number(num)
        time_lines.append(f"T{formatted}")
        current += increment
    
    print(f"Generated {len(time_lines)} time event lines")
    
    # Step 4: Write output
    print(f"Writing to {output_file}...")
    with open(output_file, 'w', encoding='utf-8') as f:
        # Write filtered and sorted lines
        for line in filtered_lines:
            f.write(line + '\n')
        
        # Append time event lines
        for line in time_lines:
            f.write(line + '\n')
    
    print(f"Done! Total lines in output: {len(filtered_lines) + len(time_lines)}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Filter, sort, and append time events to text file'
    )
    parser.add_argument('input_file', help='Input text file')
    parser.add_argument('--output_file', default='output.txt',
                        help='Output text file (default: output.txt)')
    
    args = parser.parse_args()
    
    process_file(args.input_file, args.output_file)