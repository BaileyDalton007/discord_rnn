### This script is for word vectorization training

### Be sure to create an 'input_files' directory and put all of your downloaded csv's into
### Also create an 'embed_output_files' directory for the script to store csv's

import csv

from os import listdir

def main(file):
    output_file = open('./embed_output_files/embed_output' + str(curr_file_num)+ '.csv', 'w', newline='', encoding='utf8')
    writer = csv.writer(output_file)

    with open('./input_files/' + file, 'r', encoding='utf8') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_num = 0

        # First go throught the file to find all messages sent by user
        for row in csv_reader:

            # Don't print heading
            if line_num != 0:
                msg = row[3]

                if (filter_check(msg)):
                    processed_msg = processString(msg)
                    if len(processed_msg) > 0:
                        writer.writerow([processed_msg])
            line_num += 1

    output_file.close()

# Ignores empty messages, call messages, and messages that are links/images
filters = ['http', 'Started a call', 'Pinned a message', 'Joined the server']
def filter_check(msg):
    if type(msg) != str:
        return False

    for item in filters:
        if item in msg:
            return False
    return True

def processString(txt):
    specialChars = """!#$%^&@*()./,"`~:;-_+=][}{?'1234567890"""
    for specialChar in specialChars:
        txt = txt.replace(specialChar, '')
    return txt


if __name__ == '__main__':
    input_files = listdir('./input_files')
    
    num_files = len(input_files)
    curr_file_num = 1

    for file in input_files:
        # Print progress updates
        print('Progress: ' + str(curr_file_num) + '/' + str(num_files))
        
        main(file)
        curr_file_num += 1
    
    print('Done!')