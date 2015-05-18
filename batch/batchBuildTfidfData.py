# -*- coding: utf8 -*-
import sys, os
sys.path.append("../")

from features.tfidftest import TFIDF

# tfidf_obj = TFIDF('/corpus/LJ2M/data/features/tfidf','tf1_tf2k1_tf3k1b08_idf1_idf2_idf3','LJ2M','tf3','tf2','tf1','idf1','idf2','idf3')
# tfidf_obj.set(tf2_k=1.0, tf3_k=1.0, tf3_b=0.8)
# tfidf_obj.calculate('/corpus/LJ2M/data/pkl/lj2m_wordlists')
# tfidf_obj.dump()

tfidf_obj = TFIDF('LJ40K','tf1','tf2','tf3','idf1','idf2','idf3')
tfidf_obj.set(tf2_k=1.0, tf3_k=1.0, tf3_b=0.3)
tfidf_obj.calculate('/corpus/LJ40K/data/pkl/lj40k_wordlists')
tfidf_obj.dump('/corpus/LJ40K/data/features/tfidf','0_800_tf1_tf2_tf3k1b03_idf1_idf2_idf3')

print 'finished!!'