# words1 = open("font_files/3500.txt", "r", encoding="utf-8").readlines()[0]
# hash1 = {}

# for i in range(len(words1)):
#     hash1[words1[i]] = i

# words2 = open("font_files/chinese_style.txt", "r", encoding="utf-8").readlines()
# hash2 = {}

# for i in range(len(words2)):
#     hash2[words2[i][0]] = i 

# with open("font_files/3500_style.txt", "w", encoding="utf-8") as f:
#     for i in range(len(words1)):
#         tmp1 = str(i) + '.jpg'
#         tmp2 = words2[hash2[words1[i]]][1:]
#         f.write(tmp1 + tmp2)
