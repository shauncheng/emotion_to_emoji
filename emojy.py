import numpy as np
from emo_utils import *
import emoji
from model import *
from sentences_to_indices import *
from sys import argv
'''
用法: python main.py [-t|-v|-te|-e sentence ] 
参数:  -v 可视化模型
       -t 训练模型
       -te 测试模型
       -e 测试输入的句子
       sentence: 输入的英文句子
'''

def main():
    arguments = argv

    word_to_index, index_to_word, word_to_vec_map = read_glove_vecs('./data/glove.6B.50d.txt')

    #加载数据
    X_train, Y_train = read_csv('data/train_emoji.csv')
    X_test, Y_test = read_csv('data/test_emoji.csv')
    aa = max(X_train, key=len)
    maxLen = len(max(X_train, key=len).split())

    X_train_indices = sentences_to_indices(X_train, word_to_index, maxLen)
    Y_train_oh = convert_to_one_hot(Y_train, C = 5)
    
    #加载模型
    emojy_model= Emojify_V2((maxLen,), word_to_vec_map, word_to_index)
    emojy_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    print(arguments[1])
    if arguments[1]=='-v':
        #可视化模型
        print('模型结构如下...')
        emojy_model.summary() 
    elif arguments[1]=='-t':
        print('开始训练...')
        #训练模型
        emojy_model.fit(X_train_indices, Y_train_oh, epochs = 50, batch_size = 32, shuffle=True)
        emojy_model.save_weights("weight.h5");
        print("done...")
    elif arguments[1]=='-te':
        print('测试模型...')
        #测试模型
        X_test_indices = sentences_to_indices(X_test, word_to_index, max_len = maxLen)
        Y_test_oh = convert_to_one_hot(Y_test, C = 5)
        loss, acc = emojy_model.evaluate(X_test_indices, Y_test_oh)
        print("Test accuracy = ", acc)
    elif arguments[1]=='-e':
        sentence = arguments[2]
        print('预测...',sentence)
        #情感分析预测
        x_test = np.array([sentence])
        X_test_indices = sentences_to_indices(x_test, word_to_index, maxLen)
        try:
            emojy_model.load_weights('weight.h5')
        except IOError:
            print('未找到已训练的模型')
        else:
            print(x_test[0] +' '+  label_to_emoji(np.argmax(emojy_model.predict(X_test_indices))))
    




if __name__ == '__main__':
    main()
    
