class Comment_Model:
    def __init__(self, comments):
        self.comments = comments

    def format_comments(self, comments):
        result_comments = {}
        try:
            result_comments = {comment['id']:comment for comment in comments if len(comments) > 0}
        except TypeError:
            result_comment = {}
        return result_comments

    def to_dict(self):
        return {
            'comments': self.format_comments(self.comments)
        }


non_none_comment = [{'id':1, 'text':"hello, world 1"},
                    {'id':2, 'text':"hello, world 2"},
                    {'id':3, 'text':"hello, world 3"},
                    {'id':4, 'text':"hello, world 4"},
                    {'id':5, 'text':"hello, world 5"}]
none_comment = None

model1 = Comment_Model(non_none_comment)
model2 = Comment_Model(none_comment)

print(model1.to_dict())
print(model2.to_dict())
