'''Shreyash Shrivastava'''
'''1001397477'''
#-------------------------------------------------
#-------------------------------------------------
import random
import sys
import math
import operator as op

sample = sys.argv[1]
sample = int(sample)

def bernoulli(p):
    generated_samples = []

    for x in range(sample):
        generated_samples.append(random.randint(0,1)) # Generating random samples of successes and faliures

    no_of_1s = (p*sample)
    no_of_0s = (1-p)*sample


    no_of_0s = (int)(no_of_0s)
    no_of_1s =  (int)(no_of_1s)

    distribution = []

    for x in range(no_of_1s+1):
        distribution.append(1)

    for x in range(no_of_0s+1):
        distribution.append(0)


    # distribution list maintains p% 1s and (1-p)% 0s

    random.shuffle(distribution) # Shuffles the distribution sample to eliminate bias

    bernau = []


    '''Picking random item after applying bernoulli distribution'''
    # Selects the sample from the distribution lost where chance of picking 1 is p and chance of picking 0 us (1-p)
    # The selection happens length of sample times
    for x in range(sample):
        bernau.append(random.choice(distribution))

    print bernau


def binomial(n,p):
    '''Sample size is given as sample'''
    '''Binomial distribution is the probability of exactly x success in n trials'''
    # The range of each sample element is taken as n
    # elements in the random sample range from 0 to n

    # Function to calculate nCr
    def ncr(n, r):
        r = min(r, n - r)
        numer = reduce(op.mul, xrange(n, n - r, -1), 1)
        denom = reduce(op.mul, xrange(1, r + 1), 1)
        return numer // denom

    generated_samples = []
    for x in range(sample):
        generated_samples.append((int)(random.uniform(0,n)))

    print generated_samples
    bino = []

    for x in generated_samples:
        combination =  ncr(n,x)
        succes = p**x
        fails = (1-p) ** (n-x)
        #The probability of exactly x success in n trials
        bino.append(combination*succes * fails)

    print 'The random generated x (0 < x < n) values: ', generated_samples
    print 'The binomial distribution for x', bino
    return


def geometric(p):
    '''Sample size is given as sample'''
    '''The range of sample is not provided'''
    '''Geometric distribution calculates the probability of first success on the xth trial'''
    # The range of each sample element is taken as 10
    # elements in the random sample range from 1 to 10, since the range is not specified for each sample

    generated_samples = []
    for x in range(sample):
        generated_samples.append(random.randint(1,10))

    geo = []

    for x in generated_samples:
        #The probability of the first success on the xth trial
        geo.append(((1-p)**(x-1)) * p)

    print 'The random generated x (1 < x < 10) values: ', generated_samples
    print 'The geometric distribution for x: ',geo



def neg_binomial(k,p):
    '''Sample size is given as sample'''
    '''The xth trial result in the kth success'''
    '''x can vary from k to some number, x>k'''
    # The range of each sample element is taken as 10
    # elements in the random sample range from k to 10, since the range is not specified for each sample

    generated_sample = []
    for x in range(sample):
        generated_sample.append(random.randint(k,10))

    # Function to calculate nCr
    def ncr(n, r):
        r = min(r, n - r)
        numer = reduce(op.mul, xrange(n, n - r, -1), 1)
        denom = reduce(op.mul, xrange(1, r + 1), 1)
        return numer // denom

    neg = []

    for x in generated_sample:
        combination = ncr(x-1,k-1)
        failure = (1-p) ** (x-k)
        success = p ** k
        neg.append(combination*success*failure)

    print 'The random generated x (k < x < 10) values: ', generated_sample
    print 'The negative binomial distribution for x ',neg
    return


def poisson(lambda_):
    '''Sample size is given as sample'''
    '''Number of x rare events happen in lambda_ time'''
    # The range of each sample element is taken as 10
    # elements in the random sample range from 1 to 10, since the range is not specified for each sample

    generated_sample = []
    for x in range(sample):
        generated_sample.append(random.randint(0,10))

    # Function to calculate factorial
    def fact(n):
        f = 1
        for x in range(1,n+1):
            f *= x
        return f

    poi = []

    for x in generated_sample:
        exp = math.exp(-lambda_)
        lam_power = lambda_ ** x
        ft = fact(x)
        poi.append(exp*lam_power/ft)

    print 'The random generated x (1 < x < 10) values: ', generated_sample
    print 'The poission distribution for x :', poi
    return

def arb_discrete(l1):
    for x in range(len(l1)):
        l1[x] = float(l1[x])

    inputs = []
    num  = 0
    '''Adding elements to choose from'''
    for x in range(len(l1)):
        number_of_elements = l1[x] * sample
        for y in range((int)(number_of_elements)):
            inputs.append(num)
        num +=1

    '''Shuffeling the inputs array with assigned probabilities to eliminate bia'''
    random.shuffle(inputs)

    select = []

    '''Selecting numbers for the final output'''
    for x in range(sample):
        select.append(random.choice(inputs))

    print select
    return

def uniform(a,b):
    '''Sample size is given as sample'''
    '''Uniform distribution is x/(b-a)'''
    '''Calculates P(a<x<b)'''

    a = float(a)
    b = float(b)
    generated_samples = []
    # Calcualting x in the range a < x < b
    for x in range(sample):
        generated_samples.append(random.randint(a+1,b-1))

    print generated_samples

    uni = []

    for x in generated_samples: # Calculating for each x
        f_x = 0.0
        f_x = x/(b-a)
        uni.append(f_x)

    print 'The random generated x (a < x < b) values: ',generated_samples
    print 'The uniform distribution for P(a<x<b) ',uni
    return


def exponential (lambda_):
    '''Sample size is given as sample'''
    '''Exponential dsitributon is F(x) = (1-e^(-lambda x) for x > 0'''
    # The range of each sample element is taken as 10
    # elements in the random sample range from 1 to 10, since the range is not specified for each sample
    '''Calculates P(X<x)'''
    generated_samples = []
    lambda_ = float(lambda_)

    for x in range(sample):
        generated_samples.append(random.randint(1,10))

    exp = []

    for x in generated_samples:
        exponent  = 1 - math.exp(-lambda_*x)
        exp.append(exponent)

    print 'The random generated x (1 < x < 10) values: ', generated_samples
    print 'The exponential distribution for x', exp
    return


def gamma(alpha_,lambda_):

    generated_samples = []

    for x in range(sample):
        generated_samples.append(random.randint(1,10))

    gam = []

    for x in generated_samples:
        part1 = lambda_ ** alpha_ / (math.gamma(alpha_))
        part2 = x ** (alpha_-1) * math.exp(-lambda_*x)
        gam.append(part1*part2)

    print 'The random generated x (1 < x < 10) values: ', generated_samples
    print 'The gamma distribution for x :', gam
    return

def normal(mu,sigma):
    '''Sample size is given as sample'''
    # elements in the random sample range from -2 to 2, since the range is not specified for each sample

    generated_samples = []

    for x in range(sample):
        generated_samples.append(random.uniform(-2.0,2.0))

    normal_dist = []
    distribution_func = []

    # for x in generated_samples:
    #     # part1 = 1/(sigma *(2*(math.pi)**(1/2)))
    #     # part2 = math.erf( (-(x-mu)**2) / 2*(sigma**2))
    #     z = (x -mu) / sigma
    #     normal_dist.append((1.0 + math.erf(z / math.sqrt(2.0))) / 2.0)

    for x in generated_samples:
        part1 = 1/(sigma *(2*(math.pi)**(1/2)))
        part2 = math.exp((-(x-mu)**2) / 2*(sigma**2))
        normal_dist.append(part1*part2)


    print 'The random generated x (-2.0 < x < 2.0) values: ',generated_samples
    print 'The normal distribution for P(X=x): ',normal_dist



if sys.argv[2] == 'bernoulli':
    param = sys.argv[3:]
    p = param[0]
    p = float(p)
    bernoulli(p)

if sys.argv[2] == 'binomial':
    param = sys.argv[3:]
    n = param[0]
    n = int(n)
    p = param[1]
    p = (float)(p)
    binomial(n,p)

if sys.argv[2] == 'geometric':
    param = sys.argv[3:]
    p = param[0]
    p = float(p)
    geometric(p)

if sys.argv[2] == 'neg_binomial':
    param = sys.argv[3:]
    k = param[0]
    p = param[1]
    k = (int)(k)
    p = float(p)
    neg_binomial(k,p)

if sys.argv[2] == 'poisson':
    param = sys.argv[3:]
    lam = param[0]
    lam = float(lam)
    poisson(lam)

if sys.argv[2] == 'arb_discrete':
    l1 = sys.argv[3:]
    map(float,l1)
    arb_discrete(l1)

if sys.argv[2] == 'uniform':
    param = sys.argv[3:]
    a = param[0]
    b = param[1]
    a = float(a)
    b = float(b)
    uniform(a,b)

if sys.argv[2] == 'exponential':
    param = sys.argv[3:]
    lam = param[0]
    lam = float(lam)
    exponential(lam)

if sys.argv[2] == 'gamma':
    param = sys.argv[3:]
    a = param[0]
    l = param[1]
    a = float(a)
    l = float(l)
    gamma(a,l)

if sys.argv[2] == 'normal':
    param = sys.argv[3:]
    m = param[0]
    s = param[1]
    m = float(m)
    s = float(s)
    normal(m,s)
