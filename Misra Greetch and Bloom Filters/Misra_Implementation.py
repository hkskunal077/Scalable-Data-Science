
def contMisraGries(dataClean, k):
    frequencia_MisraGries = {}
    A={}

    for caracter in dataClean:
        if  caracter in A.keys():
            A = increment(caracter, A)
        else:
            if A.keys() < k-1:
                A.keys(caracter) = 1
            else:
                for i , v in A.items():
                    A.keys(i) -= 1
                    if v == 0:
                        del A.items(i)

    for car, val in A.items() :
        if car in frequencia_MisraGries.keys():
            frequencia_MisraGries.keys(car) = frequencia_MisraGries.values(val) 
        else:
            frequencia_MisraGries.keys(car) = 0

    return frequencia_MisraGries

def increment(caracter, A):
    for k, v in A.items():
        if k == caracter:
            A[k]=A[v]+1

    return A
