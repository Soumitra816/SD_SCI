import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel

class ESCAF(nn.Module):
    def _init_(self, bert_model='bert-base-uncased', hidden_dim=768, num_classes_sd=2, num_classes_sci=2):
        super(ESCAF, self)._init_()
        
        # Document Level Encoder (BERT)
        self.bert = BertModel.from_pretrained(bert_model)
        self.doc_fc = nn.Linear(hidden_dim, hidden_dim)
        
        # Sentence Level Encoder (Transformer + Emotion)
        self.sentence_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=8), num_layers=2
        )
        self.emotion_fc = nn.Linear(hidden_dim, hidden_dim)
        
        # Contextual Attention Block (CAB)
        self.W_q = nn.Linear(hidden_dim, hidden_dim)
        self.W_k = nn.Linear(hidden_dim, hidden_dim)
        self.W_v = nn.Linear(hidden_dim, hidden_dim)
        self.attention_fc = nn.Linear(hidden_dim, hidden_dim)
        
        # Task-specific layers
        self.sd_fc = nn.Linear(hidden_dim, num_classes_sd)  # Stress Detection
        self.sci_fc = nn.Linear(hidden_dim, num_classes_sci)  # Stress Cause Identification
        
    def contextual_attention(self, sentence_embeddings, document_embedding):
        query = self.W_q(document_embedding).unsqueeze(1)  # [batch, 1, hidden_dim]
        key = self.W_k(sentence_embeddings)  # [batch, seq_len, hidden_dim]
        value = self.W_v(sentence_embeddings)  # [batch, seq_len, hidden_dim]
        
        attention_scores = torch.matmul(query, key.transpose(-2, -1)) / (key.size(-1) ** 0.5)
        attention_weights = F.softmax(attention_scores, dim=-1)
        
        context_vector = torch.matmul(attention_weights, value).squeeze(1)
        return self.attention_fc(context_vector)
    
    def forward(self, input_ids, attention_mask, sentence_embeddings, emotion_embeddings):
        # Document Encoding
        bert_output = self.bert(input_ids, attention_mask=attention_mask)
        document_embedding = bert_output.pooler_output  # CLS token output
        document_embedding = self.doc_fc(document_embedding)
        
        # Sentence Encoding
        sentence_embeddings = self.sentence_encoder(sentence_embeddings)
        emotion_embeddings = self.emotion_fc(emotion_embeddings)
        augmented_sentence_embeddings = sentence_embeddings + emotion_embeddings  # Incorporate emotion
        
        # Contextual Attention Block
        context_vector = self.contextual_attention(augmented_sentence_embeddings, document_embedding)
        
        # Task Predictions
        sd_output = self.sd_fc(document_embedding)  # Stress Detection
        sci_output = self.sci_fc(context_vector)  # Stress Cause Identification
        
        return sd_output, sci_output

# Example usage:
# dataset_path = 'DATASET'
# model = ESCAF()
# input_ids, attention_mask = torch.randint(0, 30522, (4, 128)), torch.ones(4, 128)
# sentence_embeddings, emotion_embeddings = torch.randn(4, 10, 768), torch.randn(4, 10, 768)
# sd_out, sci_out = model(input_ids, attention_mask, sentence_embeddings, emotion_embeddings)
