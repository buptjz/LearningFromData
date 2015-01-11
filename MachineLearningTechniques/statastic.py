# -*- coding: utf-8 -*-
__author__ = 'wangjz'

'''分解问题'''

need_ind = "8"

in_file = "./data/format.train"
out_file = "./data/format.train" + need_ind

f = open(in_file)
lines = f.readlines()

out = open(out_file, 'w')

for l in lines:
    itms = l.split('\t')
    if itms[0].startswith(need_ind):
        itms[0] = "1"#正样本
    else:
        itms[0] = "0"#负样本
    content = "\t".join(itms)
    out.write(content)

f.close()
out.close()


