f = open('/home/hyunsoo/inversion/DF_synthesis_LDM/deit_category_hf.txt', 'r')
label_lst = f.read().split("\n")
# print(len(label_lst))
IDX2NAME = {}
for i in range(345):
    IDX2NAME[i] = label_lst[i]