from flask_wtf import FlaskForm
from wtforms import StringField, SubmitField, PasswordField,FileField
from wtforms.validators import DataRequired, URL

class ImageUploadForm(FlaskForm):
    img_url = FileField("Upload Form", validators=[DataRequired(), URL()])
    submit = SubmitField("Create Pixel Art Image")
