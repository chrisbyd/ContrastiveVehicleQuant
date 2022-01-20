

class GPQSoftMaxNet(nn.Module):
    def __init__(self, training, intn_word=16, len_code=12, n_book=12):
        super(GPQSoftMaxNet, self).__init__()
        self.word = intn_word
        self.len_code = len_code
        self.n_book = n_book
        self.n_classes = n_classes
        self.training = training
        print("len_code is {}, and n book is {}".format(len_code,n_book))
        self.Z = torch.rand((intn_word, len_code*n_book), requires_grad=True, device = 'cuda')
        torch.nn.init.xavier_uniform_(self.Z)
        # self.Prototypes = torch.rand((n_classes, len_code*n_book), requires_grad=True, device = 'cuda')
        # self.fake_linear = torch.nn.Linear(len_code*n_book, n_classes).cuda()

    # def fake_C(self, features):
    #     return self.fake_linear(features)


    def C(self, features, Prototypes):
        x = torch.split(features, self.n_book, 1)
        y = torch.split(Prototypes, self.n_book, 0)
        for i in range(self.n_book):
            sub_res = torch.unsqueeze(torch.matmul(x[i], y[i]), 2)
            if i == 0:
                res = sub_res
            else:
                res = torch.cat((res, sub_res), 2)
        logits = torch.mean(res, 2)
        return logits

    def save(self, save_path):
        torch.save(self.Z, save_path)
        # torch.save(self.Prototypes, os.path.join('p.pt'))

    @staticmethod
    def load(pfn):
        Z = torch.load(pfn)
        # self.Prototypes = torch.load(os.path.join(pfn, 'p.pt'))
        return Z