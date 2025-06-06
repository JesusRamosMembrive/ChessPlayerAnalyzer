#include <QGuiApplication>
#include <QQmlApplicationEngine>
#include <QQmlContext>
#include "PlayerHistory.h"

int main(int argc, char *argv[])
{
    QGuiApplication app(argc, argv);

    QQmlApplicationEngine engine;
    PlayerHistory history;
    engine.rootContext()->setContextProperty("historyModel", &history);
    QObject::connect(
        &engine,
        &QQmlApplicationEngine::objectCreationFailed,
        &app,
        []() { QCoreApplication::exit(-1); },
        Qt::QueuedConnection);
    engine.loadFromModule("Desktop_GUI_QML", "Main");

    return app.exec();
}
