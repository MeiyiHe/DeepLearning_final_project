
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms



def sew_images(sing_samp):
        # sing_samp is [6, 3, 256, 306], one item is batch
        # output is the image object of all 6 pictures 'sown' together
        #############
        # A | B | C #
        # D | E | F #
        #############
        
        # return [3, 768, 612]
        
        A1 = sing_samp[0][0]
        A2 = sing_samp[0][1]
        A3 = sing_samp[0][2]

        B1 = sing_samp[1][0]
        B2 = sing_samp[1][1]
        B3 = sing_samp[1][2]

        C1 = sing_samp[2][0]
        C2 = sing_samp[2][1]
        C3 = sing_samp[2][1]

        D1 = sing_samp[3][0]
        D2 = sing_samp[3][1]
        D3 = sing_samp[3][2]

        E1 = sing_samp[4][0]
        E2 = sing_samp[4][1]
        E3 = sing_samp[4][2]

        F1 = sing_samp[5][0]
        F2 = sing_samp[5][1]
        F3 = sing_samp[5][2]

        #print("F shape {}".format(F1.shape))

        T1 = torch.cat([A1, B1, C1], 1)
        T2 = torch.cat([A2, B2, C2], 1)
        T3 = torch.cat([A3, B3, C3], 1)

        B1 = torch.cat([D1, E1, F1], 1)
        B2 = torch.cat([D2, E2, F2], 1)
        B3 = torch.cat([D3, E3, F3], 1)
        #print("T1 shape {}".format(T1.shape))

        comb1 = torch.cat([T1,B1], 0)
        comb2 = torch.cat([T2,B2], 0)
        comb3 = torch.cat([T3,B3], 0)

        #print("comb1 shape {}".format(comb1.shape)) #should be 768, 612
        comb = torch.stack([comb1, comb2, comb3])
        toImg = transforms.ToPILImage()
        result = toImg(comb) # image object [3, 768, 612]
        return result