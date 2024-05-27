from django.http import HttpResponse
from django.shortcuts import render

from home import forms
from home.models import Database, convert_to_wav

database = Database(max_audio_length=400)
database.upload_vector_to_database()


def model_form_upload(request):
    if request.method == 'POST':
        form = forms.FileUploadModelForm(request.POST, request.FILES)
        if form.is_valid():
            file = form.save()
            file_path = '/home/namdp/csdl_dpt/frontend' + file.file.url

            if file_path.split('.')[-1] != 'wav':
                file_path = convert_to_wav(file_path)

            list_relevants = database.search_similarity_audio(file_path)
            
            for i in range(len(list_relevants)):
                res = list_relevants[i]
                list_relevants[i] = "http://dynamite.io.vn:9000/csdldpt/" + res.split("data/")[1]

            return render(request, 'audio_relevants.html', {'form': form, 'list_relevants': list_relevants})

    else:
        form = forms.FileUploadModelForm()

    return render(request, 'upload_form.html', {'form': form})


def loader(request):
    return HttpResponse("")
