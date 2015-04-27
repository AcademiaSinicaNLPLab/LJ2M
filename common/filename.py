
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

def get_raw_data_filename(prefix, emotion):
     return '_'.joing([prefix, emotion, 'raw.npz'])

# def get_train_data_filename(prefix, emotion):
#      return '_'.joing([prefix, emotion, 'train.npz'])

# def get_dev_data_filename(prefix, emotion):
#      return '_'.joing([prefix, emotion, 'dev.npz'])

# def get_test_data_filename(prefix, emotion):
#      return '_'.joing([prefix, emotion, 'test.npz'])
     