from bottle import route, run, request
from tools.predictor import Predictor

predictor = Predictor(use_cache=False)


@route('/')
@route('/index')
@route('/index/')
def index():
    page = '''
    <!DOCTYPE HTML>
<html>
<head><meta charset="utf-8" /></head>
 <body>
 <form method="post" action="/prediction">
  <p><h1><b>Введите текст:</b></h1><br>
   <input name="input_text" type="text" size="100" style="height:20px">
  </p>
  <p><input type="submit" value="Отправить">
 </form>

 </body>
</html>
    '''
    return page


@route('/prediction', ['GET', 'POST'])
def prediction():
    text = request.forms.input_text
    print(text)
    result = predictor.predict(text)
    response = []
    for model_name, predicted_language in result.items():
        response.append('<h1>{}: {}</h1>'.format(model_name, predicted_language))

    page = '''
    <!DOCTYPE HTML>
    <html>
    <head><meta charset="utf-8" /></head>
    <body>
    <h1>Ответы:</h1>
    {}
    <form action="/index" method="GET">
    <button type="submit">На главную</button>
    </form>
    </body>
    </html>
    '''.format('\n'.join(response))
    return page


def main():
    run(host='localhost', port=8080)


if __name__ == '__main__':
    main()
