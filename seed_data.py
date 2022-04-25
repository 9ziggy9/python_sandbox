from random import randint

def generate_item():
    result = []
    num_attempts = 100
    while num_attempts > 0:
        num_attempts -= 1
        candidate = {'pyn_id': randint(1,100), 'board_id': randint(1,150)}
        if candidate in result:
            print("collision occurred")
            continue
        else:
            result.append(candidate)
    return result

item = generate_item()

print(item)
