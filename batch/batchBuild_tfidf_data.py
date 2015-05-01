# -*- coding: utf8 -*-
import sys, os
sys.path.append("../")

from features.tfidf import TFIDF

# tfidf_obj = TFIDF('LJ2M','tf3','idf1','tf2')
# tfidf_obj.calculate('/corpus/LJ2M/data/pkl/lj2m_wordlists')
# tfidf_obj.dump('/corpus/LJ2M/data/features/tfidf')

tfidf_obj = TFIDF('LJ40K','tf3','idf1','tf2')
tfidf_obj.set(tf2_k=1.0, tf3_k=1.0, tf3_b=0.3)
tfidf_obj.calculate('/corpus/LJ40K/data/pkl/lj40k_wordlists')
tfidf_obj.dump('/corpus/LJ40K/data/features/tfidf','tf2k1_tf3k1_tf3b0.3')

print 'finished!!'