import csv
import os
IN_PRED_FILE = '/home/icarus/kaggle/Kaggle-Fish/output/final_pred_1.csv'
SUBMIT_FILE = '/home/icarus/kaggle/Kaggle-Fish/output/submit.csv'
TEST_FILE = '/home/icarus/kaggle/Kaggle-Fish/data/input-test/test_stg1'
dict = {}
cn = 0
with open(IN_PRED_FILE, 'rb') as f_in:
    reader = csv.reader(f_in)
    for row in reader:
	cn += 1
		
        # get base image name
        base_file = row[0][0:row[0].find('-')] + '.jpg'
	if cn < 3:
	    print(base_file)
	    print(row[1])
	    print(row[8])
	p_list = row[1:]
	if len(p_list) != 8 :
	    print('invalid length')
					
        if not dict.has_key(base_file):
	    dict[base_file] = []
	dict[base_file].append(p_list)
#        else :
#            dict[base_file] = p_list
#
# sort and pick minimum NoF value
f_submit = open(SUBMIT_FILE, 'w')
f_submit.write('image,ALB,BET,DOL,LAG,NoF,OTHER,SHARK,YFT\n')
all_keys = dict.keys()
cnt = 0
for k in all_keys:
    cnt += cnt
    if k == 'img_07578.jpg':
	print dict[k]
    probs = sorted(dict[k], key = lambda l:float(l[4]))[0]
    # write to submit file
    #print('Begin to write submission file ..')
    if cnt < 3:
	print(k,probs[0], probs[7])
    pred = ['%.6f' % float(p) for p in probs]
    f_submit.write('%s,%s\n' % (k, ','.join(pred)))
# NoF
p0 = 0.490000
p1 = 0.070000
in_files = os.listdir(TEST_FILE)
for f in in_files:
    if f not in all_keys:
	probs  = [p1,p1,p1,p1,p0,p1,p1,p1]
	pred = ['%.6f' % float(p) for p in probs]
        f_submit.write('%s,%s\n' % (f, ','.join(pred)))

f_submit.close()


print('Submission file successfully generated!')
