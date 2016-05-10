with open('data/TUT-acoustic-scenes-2016-development/meta.txt','r') as fid:
    lab = [x.split('\t')[1] for x in fid.readlines()]

uni_lab = set(lab)
with open('data/TUT-acoustic-scenes-2016-development/labels.txt','w') as fid:
    for x in uni_lab:
        fid.write(x)
print uni_lab
