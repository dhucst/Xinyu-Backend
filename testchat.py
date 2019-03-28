import chat
from random import randint


def get_key_res(msg_in):
    for key in chat.keywords:
        if len(key['words']) == 1:
            for s in key['words'][0]:
                if s in msg_in:
                    idx = randint(0, len(key['reply']) - 1)
                    #   while key['reply'][idx][0] != 'text':
                    #      idx = randint(0, len(key['reply']) - 1)
                    return key['reply'][idx][1]
        else:
            flag = True
            for w in key['words']:
                t = 0
                for s in w:
                    if s in msg_in:
                        t += 1
                if t == 0:
                    flag = False
                    break
            if flag:
                idx = randint(0, len(key['reply']) - 1)
                # while key['reply'][idx][0] != 'text':
                #    idx = randint(0, len(key['reply']) - 1)
                return key['reply'][idx][1]
    return 0


def get_rep_res():
    idx = randint(0, len(chat.repeat) - 1)
    # while chat.repeat[idx][0] != 'text':
    #   idx = randint(0, len(chat.repeat) - 1)
    return chat.repeat[idx][1]


def get_hehe_res():
    idx = randint(0, len(chat.hehe) - 1)
    # while chat.hehe[idx][0] != 'text':
    #  idx = randint(0, len(chat.hehe) - 1)
    return chat.hehe[idx][1]
