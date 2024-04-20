from django.http import HttpResponse
from django.shortcuts import render

from home import forms
from home.models import Database, convert_to_wav

database = Database(max_audio_length=400)


def model_form_upload(request):
    if request.method == 'POST':
        form = forms.FileUploadModelForm(request.POST, request.FILES)
        if form.is_valid():
            file = form.save()
            file_path = '/home/namdp/csdl_dpt' + file.file.url

            if file_path.split('.')[-1] is not 'wav':
                file_path = convert_to_wav(file_path)

            list_relevants = database.search_similarity_audio(file_path)

    else:
        form = forms.FileUploadModelForm()

    return render(request, 'upload_form.html', {'form': form})


def loader(request):
    return HttpResponse("")
