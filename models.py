import torch
import torch.nn as nn

class caption_only_mlp(nn.Module):
    def __init__(self, arg, input_dim) -> None:
        super(caption_only_mlp, self).__init__()
        hidden_dim = arg['hidden_dim']
        dropout = arg['dropout']
        self.layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dim[0]),
            nn.BatchNorm1d(num_features=hidden_dim[0]),
            nn.ReLU(),
            nn.Linear(hidden_dim[0], hidden_dim[1]),
            nn.BatchNorm1d(num_features=hidden_dim[1]),
            nn.ReLU(),
            nn.Linear(hidden_dim[1], hidden_dim[2]),
            nn.BatchNorm1d(num_features=hidden_dim[2]),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim[2], hidden_dim[3]),
            nn.BatchNorm1d(num_features=hidden_dim[3]),
            nn.ReLU(),
            nn.Linear(hidden_dim[3], arg['output_dim'])
        )

    def forward(self, x):
        x = self.layer(x)
        return x

class model_caption_only(nn.Module):
    def __init__(self, arg, model_ft) -> None:
        super(model_caption_only, self).__init__()
        self.image_cnn_mlp = late_fusion_cnn_mlp(arg, model_ft)
        if arg['caption_version'] == 'short':
            self.caption_mlp = late_fusion_mlp(arg, 141+128)
        elif arg['caption_version'] == 'long':
            self.caption_mlp = late_fusion_mlp(arg, 141+768)
        elif arg['caption_version'] == 'old':
            self.caption_mlp = late_fusion_mlp(arg, 141+768)
        elif arg['caption_version'] == 'image':
            self.caption_mlp = late_fusion_mlp(arg, 141+512)
        elif arg['caption_version'] == 'image_tuned':
            self.caption_mlp = late_fusion_mlp(arg, 141+512)

    def forward(self, data_dict):
        scores = self.caption_mlp(data_dict['caption_and_user_inputs'])
        return scores

class image_only_mlp(nn.Module):
    def __init__(self, arg, input_dim) -> None:
        super(image_only_mlp, self).__init__()
        hidden_dim = arg['hidden_dim']
        dropout = arg['dropout']
        self.layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dim[0]),
            nn.BatchNorm1d(num_features=hidden_dim[0]),
            nn.ReLU(),
            nn.Linear(hidden_dim[0], hidden_dim[1]),
            nn.BatchNorm1d(num_features=hidden_dim[1]),
            nn.ReLU(),
            nn.Linear(hidden_dim[1], hidden_dim[2]),
            nn.BatchNorm1d(num_features=hidden_dim[2]),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim[2], hidden_dim[3]),
            nn.BatchNorm1d(num_features=hidden_dim[3]),
            nn.ReLU(),
            nn.Linear(hidden_dim[3], arg['output_dim'])
        )

    def forward(self, x):
        x = self.layer(x)
        return x

class image_only_cnn_mlp(nn.Module):
    def __init__(self, arg, model_ft, dropout=0.2) -> None:
        super(image_only_cnn_mlp, self).__init__()
        self.conv_layer = model_ft
        self.mlp = image_only_mlp(arg, 141+arg['img_embs_size'])

    def forward(self, user_embed, image):
        x = image
        x = self.conv_layer(x)
        user_image_embs = torch.cat((user_embed, x), 1)
        image_scores = self.mlp(user_image_embs)
        return image_scores

class model_image_only(nn.Module):
    def __init__(self, arg, model_ft) -> None:
        super(model_image_only, self).__init__()
        self.image_cnn_mlp = image_only_cnn_mlp(arg, model_ft)

    def forward(self, data_dict):
        scores = self.image_cnn_mlp(data_dict['user_input'], data_dict['image_inputs'])
        return scores

class early_fusion_cnn_mlp(nn.Module):
    def __init__(self, arg, model_ft) -> None:
        super(early_fusion_cnn_mlp, self).__init__()
        if arg['caption_version'] == 'short':
            input_dim = 2317
        elif arg['caption_version'] == 'long':
            input_dim = 2317-128+768
        output_dim = arg['output_dim']
        self.conv_layer = model_ft
        self.conv1 = nn.Sequential(
            nn.Conv2d(input_dim, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False),
            nn.BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.Conv2d(1024, 512, kernel_size=(1, 1), stride=(1, 1), bias=False),
            nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.Conv2d(512, 256, kernel_size=(1, 1), stride=(1, 1), bias=False),
            nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.Conv2d(256, 128, kernel_size=(1, 1), stride=(1, 1), bias=False),
            nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.Conv2d(128, 64, kernel_size=(1, 1), stride=(1, 1), bias=False),
            nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.Conv2d(64, 32, kernel_size=(1, 1), stride=(1, 1), bias=False),
            nn.BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True))
        self.mlp = nn.Sequential(
            nn.Linear(32*7*7, 32*5*5),
            nn.BatchNorm1d(num_features=32*5*5),
            nn.ReLU(),
            nn.Linear(32*5*5, 32*3*3),
            nn.BatchNorm1d(num_features=32*3*3),
            nn.ReLU(),
            nn.Linear(32*3*3, 32*3*3),
            nn.BatchNorm1d(num_features=32*3*3),
            nn.ReLU(),
            nn.Linear(32*3*3, 32),
            nn.BatchNorm1d(num_features=32),
            nn.ReLU(),
            nn.Linear(32, output_dim)
        )
    def forward(self, caption_and_user, image):
        x = image
        x = self.conv_layer(x)
        x = torch.cat((x, caption_and_user.unsqueeze(-1).unsqueeze(-1).repeat(1,1,x.shape[-2], x.shape[-1])),dim=1)
        x = self.conv1(x)
        x = x.reshape(x.shape[0], -1)
        x = self.mlp(x)
        return x

class model_early_fusion(nn.Module):
    def __init__(self, arg, model_ft) -> None:
        super(model_early_fusion, self).__init__()
        self.image_cnn_mlp = early_fusion_cnn_mlp(arg, model_ft)

    def forward(self, data_dict):
        scores = self.image_cnn_mlp(data_dict['caption_and_user_inputs'], data_dict['image_inputs'])
        return scores

class late_fusion_mlp(nn.Module):
    def __init__(self, arg, input_dim) -> None:
        super(late_fusion_mlp, self).__init__()
        hidden_dim = arg['hidden_dim']
        dropout = arg['dropout']
        self.layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dim[0]),
            nn.BatchNorm1d(num_features=hidden_dim[0]),
            nn.ReLU(),
            nn.Linear(hidden_dim[0], hidden_dim[1]),
            nn.BatchNorm1d(num_features=hidden_dim[1]),
            nn.ReLU(),
            nn.Linear(hidden_dim[1], hidden_dim[2]),
            nn.BatchNorm1d(num_features=hidden_dim[2]),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim[2], hidden_dim[3]),
            nn.BatchNorm1d(num_features=hidden_dim[3]),
            nn.ReLU(),
            nn.Linear(hidden_dim[3], arg['output_dim'])
        )

    def forward(self, x):
        x = self.layer(x)
        return x

class late_fusion_cnn_mlp(nn.Module):
    def __init__(self, arg, model_ft, dropout=0.2) -> None:
        super(late_fusion_cnn_mlp, self).__init__()
        self.conv_layer = model_ft
        self.mlp = late_fusion_mlp(arg, 141+arg['img_embs_size'])

    def forward(self, user_embed, image):
        x = image
        x = self.conv_layer(x)
        user_image_embs = torch.cat((user_embed, x), 1)
        image_scores = self.mlp(user_image_embs)
        return image_scores

class model_late_fusion(nn.Module):
    def __init__(self, arg, model_ft) -> None:
        super(model_late_fusion, self).__init__()
        self.image_cnn_mlp = late_fusion_cnn_mlp(arg, model_ft)
        if arg['caption_version'] == 'short':
            self.caption_mlp = late_fusion_mlp(arg, 141+128)
        elif arg['caption_version'] == 'long':
            self.caption_mlp = late_fusion_mlp(arg, 141+768)

    def forward(self, data_dict):
        image_scores = self.image_cnn_mlp(data_dict['user_input'], data_dict['image_inputs'])
        caption_scores = self.caption_mlp(data_dict['caption_and_user_inputs'])
        scores = image_scores + caption_scores
        return scores
