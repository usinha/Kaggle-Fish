import argparse
def unpickle(file):
    import cPickle
    fo = open(file, 'rb')
    dict = cPickle.load(fo)
    fo.close()
    return dict

# main
ap = argparse.ArgumentParser()
ap.add_argument("-f", "--file", required = True,
	help = "path to image")
args = vars(ap.parse_args())
file = args["file"]
dict = unpickle(file)
i = 0
super_classes = dict['coarse_labels']
print len(super_classes)
print super_classes.count(0)
#lasses.count('fish')
