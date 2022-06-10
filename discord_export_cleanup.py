### This script is for RNN training

### Be sure to create an 'input_files' directory and put all of your downloaded csv's into
### Also create an 'output_files' directory for the script to store csv's

import pandas as pd

from os import listdir

# I lost one account, so using both
# DM me if your heart desires!
USERNAMES = ['baileyD#8099', 'Plat#3996']


def main(file):
    data = pd.read_csv('./input_files/' + file, delimiter=',')
    data.reset_index()

    line_num = 0
    lines_output = 0

    output_df = pd.DataFrame(columns=['Message'])

    # Only keeping author and content columns
    data = data.drop(columns=['AuthorID', 'Date', 'Attachments', 'Reactions'])

    for _, row in data.iterrows():
        line_num += 1
        is_user = check_sender(row['Author'])

        if is_user and filter_check(row['Content']):
            new_row = {}
            msg = str(row['Content'])
            msg = processString(msg)
            msg = word_blacklist(msg)

            # Adds terminating symbol to end of each message
            new_row['Message'] = msg + " <end>"

            output_df = output_df.append(new_row, ignore_index=True)
            lines_output += 1

    output_df.to_csv('./output_files/output' + str(curr_file_num) + '.csv', index=False)
    return line_num, lines_output

# Checks if a message was sent by a user in the USERNAMES list
def check_sender(author):
    if author in USERNAMES:
        return 1
    else:
        return 0

def processString(txt):
    specialChars = """!#$%^&@*()./,"`~:;-_+=][}{?'1234567890"""
    for specialChar in specialChars:
        txt = txt.replace(specialChar, '').lower()
    return txt

def word_blacklist(txt):
    for word in blacklist:
        txt = txt.replace(word, '')
    return txt

# Ignores empty messages, call messages, and messages that are links/images
filters = ['http', 'Started a call', 'Pinned a message', 'Joined the server']
def filter_check(msg):
    if type(msg) != str:
        return False

    for item in filters:
        if item in msg:
            return False
    return True

if __name__ == '__main__':

    # Load blacklisted words from text file
    with open('blacklist.txt') as f:
        blacklist = f.read().lower().splitlines()


    input_files = listdir('./input_files')
    
    # Total line number and number of messages ouput
    total_line_num = 0
    total_out_lines = 0

    num_files = len(input_files)
    curr_file_num = 1

    for file in input_files:
        # Print progress updates
        print('Progress: ' + str(curr_file_num) + '/' + str(num_files))
        
        line_num, lines_output = main(file)

        total_line_num += line_num
        total_out_lines += lines_output

        curr_file_num += 1
    
    print(f'Done with {total_line_num} of lines checked and {total_out_lines} of lines output!')