#include "mainwindow.h"
#include "ui_mainwindow.h"


MainWindow::MainWindow(QWidget *parent)
    : QMainWindow(parent)
    , ui(new Ui::MainWindow)
{
    ui->setupUi(this);
    timerDockerStatus = new QTimer(this);
    timerDockerStatus->setSingleShot(false);
    timerDockerStatus->setInterval(3500);
    connect(timerDockerStatus, &QTimer::timeout, this, &MainWindow::displayStatusDocker, Qt::QueuedConnection);
    timerDockerStatus->start();

}

MainWindow::~MainWindow()
{
    delete ui;
}

bool MainWindow::isDockerRunning()
{
    QProcess process;
    process.start("docker", QStringList() << "info");
    if (!process.waitForFinished(3000)) {
        return false; // no respondió
    }
    int exitCode = process.exitCode();
    return (exitCode == 0);
}

void MainWindow::displayStatusDocker()
{
    if (isDockerRunning()) {
        qDebug() << "✅ Docker está levantado";
        ui->labelStatusDocker->setText( "✅ Docker está levantado");
    } else {
        qDebug() << "❌ Docker NO está levantado";
        ui->labelStatusDocker->setText( "❌ Docker NO está levantado");
    }
}


void MainWindow::on_comboBoxAction_currentIndexChanged(int index)
{
    ui->stackedWidget->setCurrentIndex(index);
}

