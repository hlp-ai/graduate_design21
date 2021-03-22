import re
def yuchuli():
    filename = 'doc1.txt'
    f = open(filename, 'r', encoding='utf-8')
    context = f.read()

    pattern = ",|\\.|\\?|!|:|;|~|，|：|。|！|；|？| "
    sentence = [i.replace('\n', '##').strip() for i in re.split(pattern, context)]

    g = open('doc2.txt', 'w', encoding='utf-8')

    for word in sentence:
        a = str(word)
        a = a.replace('##', '\r\n')
        print(a, file=g)

    f.close()
    g.close()

    k = open('doc2.txt', 'r', encoding='utf-8')
    out = open('doc3.txt', 'w', encoding='utf-8')

    for eachline in k.readlines():
        if len(eachline) > 5:
            out.writelines(eachline)
    k.close()
    out.close()


yuchuli()
