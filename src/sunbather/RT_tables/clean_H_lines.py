import pandas as pd

NIST = pd.read_table('H_lines_NIST_all.txt') #read the 'raw' file from NIST, with all (double) lines in there
NIST.index = NIST.index + 1 #shift index by 1 so that index represents the row number of the .txt file

NIST_t = NIST[NIST.term_k.notna()] #lines that have a configuration term for the upper level
NIST_n = NIST[NIST.term_k.isna()]  #lines that have only n for the upper level


double_t_lines = [] #stores the row numbers where we have an upper level with a term, but also one with just the n for that term
for i, line in NIST_t.iterrows():
    if ((NIST_n.conf_i == line.conf_i) & (NIST_n.term_i == line.term_i) & (NIST_n.J_i == line.J_i) & (line.conf_k.startswith(tuple(NIST_n.conf_k)))).any():
        double_t_lines.append(i)


with open('H_lines_NIST_all.txt', 'r') as f:
    lines = f.readlines()
with open('H_lines_NIST.txt', 'w') as f:
    for number, line in enumerate(lines):
        if number not in double_t_lines: #we remove the lines with terms if there is just a single upper n line available
            f.write(line)
        else:
            print("Removing:", line)




#this code instead filters out the upper n terms instead of J terms, but I think it's less safe in case some upper n states are not completely resolved
#double_n_lines = []
#for i, line in NIST_n.iterrows():
#    if ((NIST_t.conf_i == line.conf_i) & (NIST_t.term_i == line.term_i) & (NIST_t.J_i == line.J_i) & (NIST_t.conf_k.str.startswith(line.conf_k))).any():
#        double_n_lines.append(i)
