
import os

emotions = {}
emotions['LJ40K'] = ['accomplished', 'aggravated', 'amused', 'annoyed', 'anxious', 'awake', 'blah', 'blank', 'bored', 'bouncy', 
                    'busy', 'calm', 'cheerful', 'chipper', 'cold', 'confused', 'contemplative', 'content', 'crappy', 'crazy', 
                    'creative', 'crushed', 'depressed', 'drained', 'ecstatic', 'excited', 'exhausted', 'frustrated', 'good', 'happy', 
                    'hopeful', 'hungry', 'lonely', 'loved', 'okay', 'pissed off', 'sad', 'sick', 'sleepy', 'tired']

emotions['LJ2M'] = ['accomplished', 'aggravated', 'amused', 'angry', 'annoyed', 'anxious', 'apathetic', 'artistic', 'awake', 'bitchy', 
                    'blah', 'blank', 'bored', 'bouncy', 'busy', 'calm', 'cheerful', 'chipper', 'cold', 'complacent', 
                    'confused', 'contemplative', 'content', 'cranky', 'crappy', 'crazy', 'creative', 'crushed', 'curious', 'cynical', 
                    'depressed', 'determined', 'devious', 'dirty', 'disappointed', 'discontent', 'distressed', 'ditzy', 'dorky', 'drained', 
                    'drunk', 'ecstatic', 'embarrassed', 'energetic', 'enraged', 'enthralled', 'envious', 'exanimate', 'excited', 'exhausted', 
                    'flirty', 'frustrated', 'full', 'geeky', 'giddy', 'giggly', 'gloomy', 'good', 'grateful', 'groggy', 
                    'grumpy', 'guilty', 'happy', 'high', 'hopeful', 'horny', 'hot', 'hungry', 'hyper', 'impressed', 
                    'indescribable', 'indifferent', 'infuriated', 'intimidated', 'irate', 'irritated', 'jealous', 'jubilant', 'lazy', 'lethargic', 
                    'listless', 'lonely', 'loved', 'melancholy', 'mellow', 'mischievous', 'moody', 'morose', 'naughty', 'nauseated',
                    'nerdy', 'nervous', 'nostalgic', 'numb', 'okay', 'optimistic', 'peaceful', 'pensive', 'pessimistic', 'pissed off', 
                    'pleased', 'predatory', 'productive', 'quixotic', 'recumbent', 'refreshed', 'rejected', 'rejuvenated', 'relaxed', 'relieved', 
                    'restless', 'rushed', 'sad', 'satisfied', 'scared', 'shocked', 'sick', 'silly', 'sleepy', 'sore', 
                    'stressed', 'surprised', 'sympathetic', 'thankful', 'thirsty', 'thoughtful', 'tired', 'touched', 'uncomfortable', 'weird',
                    'working', 'worried']

emotions['LJ40K_feelingwheel'] = ['calm', 'content', 'contemplative', 'loved', 'accomplished', 'creative', 'amused', 'awake', 'cheerful', 'bouncy',
                                   'hopeful', 'excited', 'good', 'chipper', 'happy', 'ecstatic', 'crazy', 'confused', 'crappy', 'anxious', 
                                   'pissed off', 'aggravated', 'frustrated', 'annoyed', 'sleepy', 'tired', 'exhausted', 'lonely', 'cold', 'drained', 
                                   'sick', 'bored', 'crushed', 'depressed', 'hungry', 'sad', 'blah', 'blank', 'busy', 'okay']


def get_filename_by_emotion(emotion, path):
    files = os.listdir(path)
    fname = filter(lambda x: x.find(emotion) > 0, files)
    if len(fname) != 1:
        raise ValueError('scaler file name error')
    return fname[0]

def get_model_filename(emotion, c, g, ext='pkl'):
    if g is None:
        ret = 'model_%s_c%f.%s' % (emotion, c, ext)
    else:
        ret = 'model_%s_c%f_g%f.%s' % (emotion, c, g, ext)
    return ret

def get_scaler_filename(emotion, ext='pkl'):
     return 'scaler_%s.%s' % (emotion, ext)

# def get_train_data_filename(prefix, emotion):
#      return '_'.joing([prefix, emotion, 'train.npz'])

# def get_dev_data_filename(prefix, emotion):
#      return '_'.joing([prefix, emotion, 'dev.npz'])

# def get_test_data_filename(prefix, emotion):
#      return '_'.joing([prefix, emotion, 'test.npz'])
     