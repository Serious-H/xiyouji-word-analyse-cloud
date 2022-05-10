import pynlpir
from wordcloud import WordCloud
from imageio import imread
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

pynlpir.open()
#文本分词、词频统计
p = open(r'《西游记》.txt', 'r', encoding = 'utf-8')
q = open(r'test_result.txt', 'w', encoding = 'utf-8')
 #读入停用词文件
stopwords=open('stopwords.txt').read()

counts = {}      #定义空字典
fenci_word=[]  #定义空列表

#自定义词汇
pynlpir.nlpir.AddUserWord('悟空'.encode('utf8'),'noun')
pynlpir.nlpir.AddUserWord('孙悟空'.encode('utf8'),'noun')
pynlpir.nlpir.AddUserWord('唐三藏'.encode('utf8'),'noun')
pynlpir.nlpir.AddUserWord('八戒'.encode('utf8'),'noun')
pynlpir.nlpir.AddUserWord('悟能'.encode('utf8'),'noun')
pynlpir.nlpir.AddUserWord('毛脸雷公嘴'.encode('utf8'),'noun')
pynlpir.nlpir.AddUserWord('行者'.encode('utf8'),'noun')
pynlpir.nlpir.AddUserWord('孙行者'.encode('utf8'),'noun')
pynlpir.nlpir.AddUserWord('唐僧'.encode('utf8'),'noun')
pynlpir.nlpir.AddUserWord('玄奘'.encode('utf8'),'noun')
pynlpir.nlpir.AddUserWord('沙僧'.encode('utf8'),'noun')
pynlpir.nlpir.AddUserWord('悟净'.encode('utf8'),'noun')

#逐行分词 显示词性
for line in p.readlines():
    words = pynlpir.segment(line, pos_english=False)     # 把词性标注语言变更为汉语
    for word,flag in words:
        q.write(str(word) + str(flag) + " ")
        if (len(word) == 1)or(word in stopwords):
            continue
        else:
            rword = word
            s = rword+','+flag                #将分词和词性进行拼接
        counts[s] = counts.get(s,0) + 1       #同时统计分词和词性的词频，存储在字典counts中
        fenci_word.append(rword)         #将分好词的文本存到列表fenci_word里
    q.write('\n')

#字典切片函数
# def dict_slice(adict, start, end):
#     keys = adict.keys()
#     dict_slice = {}
#     for k in list(keys)[start:end]:
#         dict_slice[k] = adict[k]
#     return dict_slice

#字典输出             
#print(dict_slice(counts,0,20))
#将字典转换为列表排序后遍历输出
# items = list(counts.items())                                
# items.sort(key=lambda x:x[1],reverse = True)          
# print(items[:20])

#制作词云
font_wc=r'C:\Windows\Fonts\STXINGKA.TTF'
mytext=' '.join(fenci_word)
bg_pic=np.array(Image.open('wukong4.png'))
wc=WordCloud(font_path=font_wc,max_words=500,max_font_size=200,mask=bg_pic,background_color='white',scale=15.5)
wc.generate(mytext)
wc.to_file("wk1.png")
plt.imshow(wc,interpolation='bilinear')
plt.axis('off')
plt.show()