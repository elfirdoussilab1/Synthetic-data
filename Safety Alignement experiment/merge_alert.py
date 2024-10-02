import jsonlines
import json
import argparse

def merge_responses(original_file, response_file, output_file):
    with jsonlines.open(original_file) as f_original:
        original_data = list(f_original)
        
        with open(response_file, 'r') as f_response:
            response_data = json.loads(f_response.read().strip())
            
        if len(original_data) != len(response_data):
            raise ValueError("The number of entries in original.jsonl and the response list must match.")
        
        for original_item, response in zip(original_data, response_data):
            original_item['response'] = response.strip()
        
        with jsonlines.open(output_file, mode='w') as f_out:
            for item in original_data:
                f_out.write(item)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge original and response JSONL files.")
    parser.add_argument('--original_file', type=str, required=True, help="Path to the original JSONL file.")
    parser.add_argument('--response_file', type=str, required=True, help="Path to the response JSON file.")
    parser.add_argument('--output_file', type=str, required=True, help="Path to the output JSONL file.")

    args = parser.parse_args()

    merge_responses(args.original_file, args.response_file, args.output_file)
