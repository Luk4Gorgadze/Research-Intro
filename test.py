SymbolMapping = {
    '.' : ',',
    ',' : '.',
    '!' : '?',
    '?' : '!',
    '0' : '1',
    '1' : '0',
    '2' : '3',
    '3' : '2',
    '4' : '5',
    '5' : '4',
    '6' : '7',
    '7' : '6',
    '8' : '9',
    '9' : '8'
}
def encryptCaesarCipher(shift,text):
    lst = []
    for letter in text:
        if letter not in SymbolMapping and not letter.isalpha():
            print("1")
            lst.append(letter)
        elif letter in SymbolMapping:
            print("2")
            lst.append(SymbolMapping[letter])
        else:
            print(ord(letter))
            index = 0
            if ord(letter) <= ord("a"):
                index = ord("A") + (ord(letter) - ord("A") + shift) % 26
            if ord(letter) >= ord("a"):
                print("Here")
                index = ord("a") + (ord(letter) - ord("a") + shift) % 26
            lst.append(chr(index))   
    return ''.join(lst)

print(encryptCaesarCipher(27,"AZZZZZW"))

s = set()
s.ad