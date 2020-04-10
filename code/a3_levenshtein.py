import os
import numpy as np

dataDir = '/u/cs401/A3/data/'
#dataDir = "C:\\Users\\LAI\\Desktop\\CSC401\\Speech_Recognition\\data"

def Levenshtein(r, h):
    """                                                                         
    Calculation of WER with Levenshtein distance.                               
                                                                                
    Works only for iterables up to 254 elements (uint8).                        
    O(nm) time ans space complexity.                                            
                                                                                
    Parameters                                                                  
    ----------                                                                  
    r : list of strings                                                                    
    h : list of strings                                                                   
                                                                                
    Returns                                                                     
    -------                                                                     
    (WER, nS, nI, nD): (float, int, int, int) WER, number of substitutions, insertions, and deletions respectively
                                                                                
    Examples                                                                    
    --------                                                                    
    >>> wer("who is there".split(), "is there".split())                         
    0.333 0 0 1                                                                           
    >>> wer("who is there".split(), "".split())                                 
    1.0 0 0 3                                                                           
    >>> wer("".split(), "who is there".split())                                 
    Inf 0 3 0                                                                           
    """
    n = len(r)
    m = len(h)
    r_len = len(r) + 1
    h_len = len(h) + 1
    R = np.zeros((r_len, h_len))
    B = np.zeros((r_len, h_len, 3)) #Count no of insertion, deletion, and substitution 
    for i in range(r_len):
        R[i, 0] = i
    for j in range(h_len):
        R[0, j] = j
    
    for i in range(1, r_len):
        for j in range( 1, h_len):
            deletion = R[i - 1, j] + 1
            #If words match
            if r[i - 1] == h[j - 1]:
                substitution = R[i - 1, j - 1]
            else:
                substitution = R[i - 1, j - 1] + 1
            insertion = R[i, j - 1] + 1
            R[i , j] = min([deletion, substitution, insertion])
            #Priority Order
            #0 Substitution
            #1 Insertion
            #2 Deleteion
            if R[i, j] == substitution:
                #Add One substituion count
                B[i, j, 0] = B[i - 1, j - 1, 0] + 1
                B[i, j, 1] = B[i - 1, j - 1, 1]
                B[i, j, 2] = B[i - 1, j - 1, 2]
            elif R[i, j] == insertion:
                #Add One Insertion count
                B[i, j, 0] = B[i, j - 1, 0]
                B[i, j, 1] = B[i, j - 1, 1] + 1
                B[i, j, 2] = B[i, j - 1, 2]
            else:
                #Add One Deletion count
                B[i, j, 0] = B[i - 1, j, 0]
                B[i, j, 1] = B[i - 1, j, 1]
                B[i, j, 2] = B[i - 1, j, 2] + 1
    wer = R[n,m] / n
    nS = B[n,m,0]
    nI = B[n,m,1]
    nD = B[n,m,2]
    return wer, nS, nI, nD

def preproc(text):
    preprocText = text.strip()
    punctuations = '!"#$%&\'()*+,-./:;<=>?@\\^_`{|}~'
    preprocText = preprocText.strip(punctuations).lower()
    return preprocText.split()
    
if __name__ == "__main__":
    google_wers = []
    kaldi_wers = []
    f = open("asrDiscussion.txt","w+")
    for subdir, dirs, files in os.walk(dataDir):
        for speaker in dirs:
            print(speaker)
            gold_standard_lines = open(os.path.join(dataDir, speaker, 'transcripts.txt'), 'r').read().splitlines()
            kaldi_lines = open(os.path.join(dataDir, speaker, 'transcripts.Kaldi.txt'), 'r').read().splitlines()
            google_lines = open(os.path.join(dataDir, speaker, 'transcripts.Google.txt'), 'r').read().splitlines()

            for i in range(len(gold_standard_lines)):
                #ignore such cases where one of the 3 transcript files are missing as suggested in Piazza
                if (gold_standard_lines[i] == "" or kaldi_lines[i] == "" or google_lines[i] == ""):
                    continue
                #Preprocess the text
                gold_standard = preproc(gold_standard_lines[i])
                kaldi = preproc(kaldi_lines[i])
                google = preproc(google_lines[i])
                kaldi_wer, kaldi_nS, kaldi_nI, kaldi_nD = Levenshtein(gold_standard, kaldi)
                google_wer, google_nS, google_nI, google_nD = Levenshtein(gold_standard, google)
                f.write('{} {} {} {} S:{}, I:{}, D:{}\n'.format(speaker, 'Google', i, google_wer, google_nS, google_nI, google_nD))
                f.write('{} {} {} {} S:{}, I:{}, D:{}\n'.format(speaker, 'Kaldi', i, kaldi_wer, kaldi_nS, kaldi_nI, kaldi_nD))
                google_wers.append(google_wer)
                kaldi_wers.append(kaldi_wer)
        google_stats = np.array(google_wers)
        kaldi_stats = np.array(kaldi_wers)
    f.write('Google Mean: {}, Kaldi Mean: {}, Google std: {}, Kaldi std: {}\n'.format(google_stats.mean(), kaldi_stats.mean(), google_stats.std(), kaldi_stats.std()))
    f.close()