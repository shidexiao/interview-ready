import hashlib

def hash_do(x,y):
    return hashlib.md5(x.encode()).hexdigest()+'_'+hashlib.md5(y.encode()).hexdigest()


if __name__ == '__main__':
    res = hash_do('110104196001010029','13800000001')
    print(res)