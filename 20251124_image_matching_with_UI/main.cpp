#include "_image_matching_with_UI.h"
#include <QtWidgets/QApplication>

int main(int argc, char *argv[])
{
    QApplication app(argc, argv);
    _image_matching_with_UI window;
    window.show();
    return app.exec();
}
