### This script is for word vectorization training

### Be sure to create an 'input_files' directory and put all of your downloaded csv's into
### Also create an 'embed_output_files' directory for the script to store csv's
### Create a blacklist.txt file to remove words from the data set

import csv

from os import listdir
import string

# Having messages shorter than the window in the w2v model does not make sense, so filter out
# messages that have a length less than MIN_MSG_LENGTH words
MIN_MSG_LENGTH = 5

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
                msg = row[3].lower()

                if (filter_check(msg)):
                    processed_msg = process_string(msg)
                    processed_msg = word_blacklist(processed_msg)

                    # Check word count of message
                    word_count = sum([i.strip(string.punctuation).isalpha() for i in processed_msg.split()])
                    if word_count >= MIN_MSG_LENGTH:
                        writer.writerow([processed_msg])
            line_num += 1

    output_file.close()

# Ignores empty messages, call messages, and messages that are links/images
filters = ['http', 'started a call', 'pinned a message', 'joined the server']
def filter_check(msg):
    if type(msg) != str:
        return False

    for item in filters:
        if item in msg:
            return False
    return True

def process_string(txt):
    special_chars = """!#$%^&@*()./,"`~:;-_+=][}{?'1234567890"""
    for special_char in special_chars:
        txt = txt.replace(special_char, '')
    return txt

def word_blacklist(txt):
    for word in blacklist:
        txt = txt.replace(word, '')
    return txt
    

if __name__ == '__main__':

    # Load blacklisted words from text file
    with open('blacklist.txt') as f:
        blacklist = f.read().lower().splitlines()


    input_files = listdir('./input_files')
    
    num_files = len(input_files)
    curr_file_num = 1

    for file in input_files:
        # Print progress updates
        print('Progress: ' + str(curr_file_num) + '/' + str(num_files))
        
        main(file)
        curr_file_num += 1
    
    print('Done!')