#arg1 is term1

#arg2 is term2

#arg3 is the way to get the score, valid input including
#all, max, topfive, sum
#[SymbolOf, CreatedBy, MadeOf, PartOf, HasLastSubevent, HasFirstSubevent, Desires, CausesDesire,
#DefinedAs, HasA, ReceivesAction, MotivatedByGoal, Causes, HasProperty, HasPrerequisite,
#HasSubevent, AtLocation, IsA, CapableOf, UsedFor]
#case insensitive for the third argument
import pickle
import numpy as np
import sys
import math

def getVec(We,words,t):
    t = t.strip()
    array = t.split('_')
    if array[0] in words:
        vec = We[words[array[0]],:]
    else:
        vec = We[words['UUUNKKK'],:]
        # print 'can not find corresponding vector:',array[0].lower()
    for i in range(len(array)-1):
        if array[i+1] in words:
            vec = vec + We[words[array[i+1]],:]
        else:
            # print 'can not find corresponding vector:',array[i+1].lower()
            vec = vec + We[words['UUUNKKK'],:]
    vec = vec/len(array)
    return vec

def sigmoid(x):
    return 1 / (1 + math.exp(-x))

def score(term1,term2,words,We,rel,Rel,Weight,Offset,evaType):
    # print(rel)
    v1 = getVec(We,words,term1)
    v2 = getVec(We,words,term2)
    result = {}

    # del_rels = ['HasPainIntensity','HasPainCharacter','LocationOfAction','LocatedNear',
    # 'DesireOf','NotMadeOf','InheritsFrom','InstanceOf','RelatedTo','NotDesires',
    # 'NotHasA','NotIsA','NotHasProperty','NotCapableOf']
    #
    # for del_rel in del_rels:
    #     del rel[del_rel.lower()]

    for k,v in rel.items():
        v_r = Rel[rel[k],:]
        gv1 = np.tanh(np.dot(v1,Weight)+Offset)
        gv2= np.tanh(np.dot(v2,Weight)+Offset)

        temp1 = np.dot(gv1, v_r)
        score = np.inner(temp1,gv2)
        result[k] = (sigmoid(score))

    if(evaType.lower()=='max'):
        result = sorted(result.items(), key=lambda x: x[1], reverse = True)
        # for k,v in result[:1]:
        #     print k, 'score:', v
        return result[:1]
    if(evaType.lower()=='topfive'):
        result = sorted(result.items(), key=lambda x: x[1], reverse = True)
        for k,v in result[:5]:
          print k, 'score:', v
        return result[:5]
    if(evaType.lower()=='sum'):
        result = sorted(result.items(), key=lambda x: x[1], reverse = True)
        total = 0
        for i in result:
            total = total + i[1]
        print 'total score is:',total
        return total
    if(evaType.lower()=='all'):
        result = sorted(result.items(), key=lambda x: x[1], reverse = True)
        for k,v in result[:]:
          print k, 'score:', v
        return result
    else:
        tar_rel = evaType.lower()
        if result.get(tar_rel) == None:
            print 'illegal relation, please re-enter a valid relation'
            return 'None'
        else:
            print tar_rel,'relation score:',result.get(tar_rel)
            return result.get(tar_rel)


if __name__ == "__main__":
    model = pickle.load(open("Bilinear_cetrainSize300frac1.0dSize200relSize150acti0.001.1e-05.800.RAND.tanh.txt19.pickle", "r"))

    Rel = model['rel']
    print(Rel.shape)
    We = model['embeddings']
    print(We.shape)
    Weight = model['weight']
    print(Weight.shape)
    Offset = model['bias']
    words = model['words_name']
    rel = model['rel_name']
    print(rel)

    result = score(str(sys.argv[1]),str(sys.argv[2]),words,We,rel,Rel,Weight,Offset,str(sys.argv[3]))
