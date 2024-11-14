import random

# Function to insert a character at a specified position in a string
def insert(s, pos, ch):
    return s[:pos] + ch + s[pos:]

# Function to add more characters to the string to meet the required conditions
def add_more_char(s, need):
    pos = 0
    low_case = "abcdefghijklmnopqrstuvwxyz"
    for i in range(need):
        pos = random.randint(0, len(s) - 1)
        s = insert(s, pos, low_case[random.randint(0, 25)])
    return s

# Function to suggest a new password by adding missing character types
def suggester(l, u, d, s, st):
    num = '0123456789'
    low_case = "abcdefghijklmnopqrstuvwxyz"
    up_case = low_case.upper()
    spl_char = '@#$_()!'
    pos = 0

    # Add a lowercase letter if missing
    if l == 0:
        pos = random.randint(0, len(st) - 1)
        st = insert(st, pos, low_case[random.randint(0, 25)])

    # Add an uppercase letter if missing
    if u == 0:
        pos = random.randint(0, len(st) - 1)
        st = insert(st, pos, up_case[random.randint(0, 25)])

    # Add a digit if missing
    if d == 0:
        pos = random.randint(0, len(st) - 1)
        st = insert(st, pos, num[random.randint(0, 9)])

    # Add a special character if missing
    if s == 0:
        pos = random.randint(0, len(st) - 1)
        st = insert(st, pos, spl_char[random.randint(0, len(spl_char) - 1)])

    return st

# Function to generate a password that satisfies the strength criteria
def generate_password(n, p):
    l = u = d = s = 0
    need = 0

    # Check if the password contains at least one lowercase, uppercase, digit, and special character
    for i in range(n):
        if p[i].islower():
            l = 1
        elif p[i].isupper():
            u = 1
        elif p[i].isdigit():
            d = 1
        else:
            s = 1

    # If the password contains all required character types, it's strong
    if (l + u + d + s) == 4:
        print("Your Password is Strong")
        return
    else:
        print("Suggested Passwords:")
        for i in range(10):
            suggest = suggester(l, u, d, s, p)
            need = 8 - len(suggest)  # Ensure the password length is at least 8 characters
            if need > 0:
                suggest = add_more_char(suggest, need)
            print(suggest)

if __name__ == '__main__':
    input_string = input("Enter your password: ")
    generate_password(len(input_string), input_string)
