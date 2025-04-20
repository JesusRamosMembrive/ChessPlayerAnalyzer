import QtQuick 2.15
import QtQuick.Controls 2.15
import QtQuick.Layouts 1.15

Page {
    id: results
    property string username: ""
    anchors.fill: parent
    background: Rectangle { color: "transparent" }

    header: ToolBar {
        contentItem: RowLayout {
            anchors.fill: parent
            spacing: 12
            ToolButton {
                text: "\u2190"
                onClicked: StackView.view.pop()
                background: Rectangle { color: "transparent" }
            }
            Label {
                text: results.username
                color: "white"
                font.pixelSize: 20
            }
        }
    }

    RowLayout {
        anchors.fill: parent
        anchors.margins: 24
        spacing: 24

        ColumnLayout {
            id: metricsPanel
            Layout.fillWidth: true
            Layout.preferredWidth: parent.width * 0.6
            spacing: 16

            MetricCard { title: qsTr("Entropía de aperturas") }
            MetricCard { title: qsTr("Tiempo entre jugadas") }
            MetricCard { title: qsTr("Partidas analizadas") }
            MetricCard { title: qsTr("Pausas largas") }
        }

        Rectangle {
            id: boardPanel
            Layout.preferredWidth: parent.width * 0.35
            Layout.fillHeight: true
            color: "#1e1e1e"
            radius: 8
            Text {
                anchors.centerIn: parent
                text: qsTr("Chessboard placeholder")
                color: "#505050"
                font.pixelSize: 16
            }
        }
    }
}
