#!/usr/bin/python

from nltk.stem.snowball import SnowballStemmer
import string

def parseOutText(f,working = False):
    """ given an opened email file f, parse out all text below the
        metadata block at the top
        (in Part 2, you will also add stemming capabilities)
        and return a string that contains all the words
        in the email (space-separated) 
        
        example use case:
        f = open("email_file_name.txt", "r")
        text = parseOutText(f)
        
        working 显示处于测试模式还是工作模式
        在工作模式，f.read()可能得到的是bytes，需要解码以后得到其中的文本
        """


    f.seek(0)  ### go back to beginning of file (annoying)
    all_text = f.read()
    
    if working == True:
        all_text = all_text.decode('utf-8')
    
    ### split off metadata
    content = all_text.split("X-FileName:")
    words = ""
    if len(content) > 1:
        ### remove punctuation
        '''
        maketrans生成一个字符转换表，用来将第一参数的字符转换为第二参数。第三参数是要删除的
        字符。
        而translate执行这个字符操作。
        '''
        ### project part 2: comment out the line below
        #words = text_string

        ### split the text string into individual words, stem each word,
        text_string = content[1].translate(str.maketrans("", "", string.punctuation))
        ### and append the stemmed word to words (make sure there's a single
        text_string = text_string.split()

        stemmer = SnowballStemmer("english", ignore_stopwords=False)
        for counter in range(len(text_string)):
            text_string[counter] = stemmer.stem(text_string[counter])
            
        ### space between each stemmed word)
        words = " ".join(text_string)


    return words

    

def main():
    ff = open("../text_learning/test_email.txt", "r")
    text = parseOutText(ff)
    print(text)



if __name__ == '__main__':
    main()

