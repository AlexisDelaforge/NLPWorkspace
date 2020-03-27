# Own modules to handle dictionary

# My code


def dict_less(dictionary, argument_to_forget):
    new_dictionary = dict(dictionary)
    for argument in argument_to_forget:
        try:
            new_dictionary.pop(argument)
        except KeyError:
            print('KeyError in argument to forget in dict_less function')
    return new_dictionary


# My code


def dict_change(dictionary, argument_to_update):
    new_dictionary = dict(dictionary)
    for key, value in argument_to_update.items():
        try:
            new_dictionary[key] = value
        except KeyError:
            print('KeyError in argument to update in dict_change function')
    return new_dictionary


# My code

def split_values(dataset_length, listed_len):
    total_listed = 0
    for i in listed_len:
        if i == (len(listed_len) - 1):
            listed_len[i] = len(dataset_length) - total_listed
        else:
            listed_len[i] = round(listed_len[i] * len(dataset_length))
            total_listed += listed_len[i]
    return listed_len
