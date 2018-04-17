# -*- coding: utf-8 -*-
"""
Created on Tue Apr  3 16:54:58 2018

@author: Bird

这个文件生成一个tar接口，以便程序可以直接访问压缩包
"""
import tarfile
tar = tarfile.open('../maildir/enron_mail_20150507.tar.gz','r:gz')

#temp = tar.getnames()

f = tar.extractfile('maildir/germany-c/_sent_mail/1.')

text = f.read()