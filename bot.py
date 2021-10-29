import telebot
import requests
import io
from PIL import Image
from telebot import types
from config import TOKEN, HOST, PORT

bot = telebot.TeleBot(TOKEN)

user_steps = {}
known_users = []
ml_info = {}
ml_update = ''
model_dic = {'Регрессия': {'Линейная модель': 'Ridge',
                           'Метод опорных векторов': 'SVR',
                           'Дерево решений': 'DecisionTreeRegressor',
                           'Случайный лес': 'RandomForestRegressor'},
             'Классификация': {'Линейная модель': 'LogisticRegression',
                               'Метод опорных векторов': 'SVC',
                               'Дерево решений': 'DecisionTreeClassifier',
                               'Случайный лес': 'RandomForestClassifier'}}
commands = {'start': 'Запустить бота',
            'model': 'Обучить ML-модель на данных',
            'predict': 'Получить предсказания модели',
            'retrain': 'Обучить модель заново',
            'delete': 'Удалить модель',
            'ml_models': 'Посмотреть список доступных для обучения классов моделей',
            'all_models': 'Посмотреть словарь всех обученных моделей',
            'plot': 'Построить графики для модели',
            'help': 'Получить справку по меню с командами'}


def get_user_step(uid):
    if uid in user_steps:
        return user_steps[uid]
    else:
        known_users.append(uid)
        user_steps[uid] = 0
        return user_steps[uid]


@bot.message_handler(commands=['help'])
def help_command_handler(message):
    cid = message.chat.id
    help_text = 'Доступны следующие команды: \n'
    for key in commands:
        help_text += '/' + key + ': '
        help_text += commands[key] + '\n'
    bot.send_message(cid, help_text)


@bot.message_handler(commands=['start'])
def start_command_handler(message):
    cid = message.chat.id
    user_steps[cid] = -1
    bot.send_message(cid, 'Привет! Выбери команду из списка')
    help_command_handler(message)


@bot.message_handler(commands=['ml_models'])
def ml_models_command_handler(message):
    cid = message.chat.id
    text = 'Доступны следующие модели для обучения: \n'
    response = requests.get(f'http://{HOST}:{PORT}/ml_api').json()
    for resp in response:
        text += resp + '\n'
    bot.send_message(cid, text)


@bot.message_handler(commands=['all_models'])
def all_models_command_handler(message):
    cid = message.chat.id
    response = requests.get(f'http://{HOST}:{PORT}/ml_api/all_models').json()
    text = f'Всего было обучено моделей: {len(response)} \n'
    for resp in response:
        text += resp + ' – ' + str(response[resp]) + '\n'
    bot.send_message(cid, text)


@bot.message_handler(commands=['model'])
def model_command_handler(message):
    global ml_info
    ml_info = {'params': 'default',
               'grid_search': False,
               'param_grid': 'default'}
    cid = message.chat.id
    markup = types.ReplyKeyboardMarkup(row_width=2, resize_keyboard=True)
    button_reg = types.KeyboardButton(text='Регрессия')
    button_clf = types.KeyboardButton(text='Классификация')
    markup.add(button_reg, button_clf)
    user_steps[cid] = 0
    bot.send_message(cid, 'Выбери задачу, которую хочешь решить', reply_markup=markup)


@bot.message_handler(func=lambda message: get_user_step(message.chat.id) == 0)
def get_model(message):
    ml_info['task'] = message.text
    cid = message.chat.id
    markup = types.ReplyKeyboardMarkup(row_width=2, resize_keyboard=True)
    button_lin = types.KeyboardButton(text='Линейная модель')
    button_svm = types.KeyboardButton(text='Метод опорных векторов')
    button_tree = types.KeyboardButton(text='Дерево решений')
    button_rf = types.KeyboardButton(text='Случайный лес')
    markup.add(button_lin, button_svm, button_tree, button_rf)
    user_steps[cid] = 1
    bot.send_message(cid, 'Выбери модель, которую хочешь построить', reply_markup=markup)


@bot.message_handler(func=lambda message: get_user_step(message.chat.id) == 1)
def get_data(message):
    ml_info['model'] = model_dic[ml_info['task']][message.text]
    del ml_info['task']
    cid = message.chat.id
    user_steps[cid] = 2
    bot.send_message(cid, 'Загрузи данные в формате json')


@bot.message_handler(func=lambda message: get_user_step(message.chat.id) == 2, content_types=['document'])
def get_params(message):
    file_id = message.document.file_id
    downloaded_file = requests.get(bot.get_file_url(file_id))
    ml_info['data'] = downloaded_file.text
    cid = message.chat.id
    # bot.send_message(cid, bot.get_file_url(file_id))
    markup = types.ReplyKeyboardMarkup(row_width=2, resize_keyboard=True)
    button_yes = types.KeyboardButton(text='Да')
    button_no = types.KeyboardButton(text='Нет')
    markup.add(button_yes, button_no)
    user_steps[cid] = 3
    bot.send_message(cid, 'Использовать гиперпараметры по умолчанию?', reply_markup=markup)


@bot.message_handler(func=lambda message: get_user_step(message.chat.id) == 3 and message.text == 'Нет')
def get_user_params(message):
    cid = message.chat.id
    user_steps[cid] = 4
    bot.send_message(cid, 'Введи значения гиперпараметров через запятую')


@bot.message_handler(func=lambda message: get_user_step(message.chat.id) == 4)
def save_user_params(message):
    cid = message.chat.id
    ml_info['params'] = {}
    for param_value in message.text.split(','):
        param = param_value.split('=')[0].strip()
        try:
            value = float(param_value.split('=')[1].strip())
        except ValueError:
            value = param_value.split('=')[1].strip()
        ml_info['params'][param] = value
    user_steps[cid] = 3


@bot.message_handler(func=lambda message: get_user_step(message.chat.id) == 3)
def get_grid_search(message):
    cid = message.chat.id
    markup = types.ReplyKeyboardMarkup(row_width=2, resize_keyboard=True)
    button_yes = types.KeyboardButton(text='Да')
    button_no = types.KeyboardButton(text='Нет')
    markup.add(button_yes, button_no)
    user_steps[cid] = 5
    bot.send_message(cid, 'Подбирать оптимальные значения гиперпараметров?', reply_markup=markup)


@bot.message_handler(func=lambda message: get_user_step(message.chat.id) == 5 and message.text == 'Да')
def get_param_grid(message):
    cid = message.chat.id
    ml_info['grid_search'] = True
    markup = types.ReplyKeyboardMarkup(row_width=2, resize_keyboard=True)
    button_yes = types.KeyboardButton(text='Да')
    button_no = types.KeyboardButton(text='Нет')
    markup.add(button_yes, button_no)
    user_steps[cid] = 6
    bot.send_message(cid, 'Использовать дефолтную сетку для перебора?', reply_markup=markup)


@bot.message_handler(func=lambda message: get_user_step(message.chat.id) == 6 and message.text == 'Нет')
def get_user_param_grid(message):
    cid = message.chat.id
    user_steps[cid] = 7
    bot.send_message(cid, 'Введи значения гиперпараметров для перебора через точку с запятой')


@bot.message_handler(func=lambda message: get_user_step(message.chat.id) == 7)
def save_param_grid(message):
    cid = message.chat.id
    ml_info['param_grid'] = {}
    for params in message.text.split(';'):
        for param_values in params.split(','):
            param = param_values.split('=')[0].strip()
            values = param_values.split('=')[1].split(',')
            try:
                list_values = [float(i) for i in values]
            except ValueError:
                list_values = [i.strip() for i in values]
            ml_info['param_grid'][param] = list_values
    user_steps[cid] = 5


@bot.message_handler(func=lambda message: get_user_step(message.chat.id) == 5)
def fit_model(message):
    cid = message.chat.id
    response = requests.post(f'http://{HOST}:{PORT}/ml_api', json=ml_info)
    user_steps[cid] = -1
    if response.status_code == 200:
        bot.send_message(cid, 'Обучение модели завершено!')
    else:
        bot.send_message(cid, 'Что-то пошло не так:( Попробуй начать заново')


@bot.message_handler(commands=['predict'])
def predict_command_handler(message):
    cid = message.chat.id
    user_steps[cid] = 8
    bot.send_message(cid, 'Введи id модели, для которой хочешь получить предсказания')


@bot.message_handler(func=lambda message: get_user_step(message.chat.id) == 8)
def return_predictions_command_handler(message):
    cid = message.chat.id
    user_steps[cid] = -1
    response = requests.get(f'http://{HOST}:{PORT}/ml_api/{message.text}').json()
    text = 'Train predictions: ' + response['train_predictions'] + '\n'
    text += 'Test predictions: ' + response['test_predictions'] + '\n'
    bot.send_message(cid, text)


@bot.message_handler(commands=['retrain'])
def retrain_command_handler(message):
    cid = message.chat.id
    user_steps[cid] = 9
    bot.send_message(cid, 'Введи id модели, которую хочешь обучить заново')


@bot.message_handler(func=lambda message: get_user_step(message.chat.id) == 9)
def retrain_data_command_handler(message):
    cid = message.chat.id
    global ml_update
    ml_update = message.text
    user_steps[cid] = 10
    bot.send_message(cid, 'Загрузи новые данные в формате json')


@bot.message_handler(func=lambda message: get_user_step(message.chat.id) == 10, content_types=['document'])
def retrain_model_command_handler(message):
    file_id = message.document.file_id
    downloaded_file = requests.get(bot.get_file_url(file_id))
    cid = message.chat.id
    user_steps[cid] = -1
    response = requests.put(f'http://{HOST}:{PORT}/ml_api/{ml_update}', json=downloaded_file.text)
    if response.status_code == 200:
        bot.send_message(cid, 'Модель переобучена!')
    else:
        bot.send_message(cid, 'Что-то пошло не так:( Попробуй еще раз')


@bot.message_handler(commands=['delete'])
def delete_command_handler(message):
    cid = message.chat.id
    user_steps[cid] = 11
    bot.send_message(cid, 'Введи id модели, которую хочешь удалить')


@bot.message_handler(func=lambda message: get_user_step(message.chat.id) == 11)
def delete_model_command_handler(message):
    cid = message.chat.id
    response = requests.delete(f'http://{HOST}:{PORT}/ml_api/{message.text}')
    if response.status_code == 204:
        bot.send_message(cid, 'Модель удалена!')
    else:
        bot.send_message(cid, 'Что-то пошло не так:( Попробуй еще раз')
    user_steps[cid] = -1


@bot.message_handler(commands=['plot'])
def plot_command_handler(message):
    cid = message.chat.id
    user_steps[cid] = 12
    bot.send_message(cid, 'Введи id модели, для которой хочешь построить графики')


@bot.message_handler(func=lambda message: get_user_step(message.chat.id) == 12)
def plot_model_command_handler(message):
    cid = message.chat.id
    user_steps[cid] = -1
    response = requests.get(f'http://{HOST}:{PORT}/ml_api/plot/{message.text}')
    img = Image.open(io.BytesIO(response.content))
    bot.send_photo(cid, img)

    response = requests.get(f'http://{HOST}:{PORT}/ml_api/plot_feature_importance/{message.text}')
    if response.status_code == 200:
        img = Image.open(io.BytesIO(response.content))
        bot.send_photo(cid, img)


if __name__ == '__main__':
    bot.polling(none_stop=True, interval=0)
