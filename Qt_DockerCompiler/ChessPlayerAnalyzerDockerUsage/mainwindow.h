#pragma

#include <QMainWindow>
#include <QCoreApplication>
#include <QProcess>
#include <QString>
#include <QDebug>
#include <QTimer>

QT_BEGIN_NAMESPACE
namespace Ui {
class MainWindow;
}
QT_END_NAMESPACE



class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    MainWindow(QWidget *parent = nullptr);
    ~MainWindow();

private:
    Ui::MainWindow *ui;
    QTimer * timerDockerStatus;
    bool isDockerRunning();

private slots:

    void displayStatusDocker();
    void on_comboBoxAction_currentIndexChanged(int index);

signals:


};
