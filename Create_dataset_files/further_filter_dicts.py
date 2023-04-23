import pickle

dict_path = '/home/luis/Dropbox/DATASETS_AUDIO/AOLME_Eng_fundamental_May09_2022/dicts/AOLME_Bilingualv2_testset_dict.pickle'
dict_words = pickle.load(open(dict_path, "rb" ))

change_ph_dict = {'oː':'o',
                  'n̩':'n',
                  'ɹ':'ɾ',
                  'ʔ':'a',
                  'ɔː':'o',
                  'uː':'u',
                  'iː':'i',
                  'ɐ':'a',
                  'ʌ':'a',
                  'ŋ':'n',
                  'ɚ':'ɾ',
                  'x':'h',
                  'əl':'l',
                  'æ':'a',
                  'ɔ':'o',
                  'ɔːɹ':'o ɾ',
                  'ɑːɹ':'o ɾ',
                  'aɪɚ':'aɪ ɾ',
                  'aɪə':'aɪ ə',
                  'ɪɹ':'ɪ ɾ',
                  'ɛɹ':'ɛ ɾ',
                  'ʊɹ':'ʊ ɾ',
                  'oːɹ':'o ɾ'}
                  # '':'',
                  # '':'',
                  # '':''}


base_dict_path = '/'.join(dict_path.split('/')[0:-1])
dict_name = dict_path.split('/')[-1].split('.')[0]


dict_txt = base_dict_path + '/{}_reduced.txt'.format(dict_name)
dict_pickle = base_dict_path + '/{}_reduced.pickle'.format(dict_name)

new_dict_file = open(dict_txt, 'w')
corrected_dictionary = {}

for current_word in dict_words:
    current_phs = dict_words[current_word]
    print(f'in  : {current_phs}')
    
    current_phs_list = current_phs.split(' ')
    new_phs_list = []
    
    for single_ph in current_phs_list:
        if single_ph in change_ph_dict.keys():
            single_ph = change_ph_dict[single_ph]
        new_phs_list.append(single_ph)

    corrected_phs = ' '.join(new_phs_list)
    print(f'out : {corrected_phs}')
    
    corrected_dictionary[current_word] = corrected_phs
    new_dict_file.write(current_word + '\t' + corrected_phs + '\n')
    
    print('---------------------------')
    
new_dict_file.close()    
pickle.dump(corrected_dictionary, open(dict_pickle, "wb"))
        