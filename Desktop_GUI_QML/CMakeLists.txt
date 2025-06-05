cmake_minimum_required(VERSION 3.16)

project(Desktop_GUI_QML VERSION 0.1 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Qt6 REQUIRED COMPONENTS Quick)

qt_standard_project_setup(REQUIRES 6.8)

qt_add_executable(appDesktop_GUI_QML
    main.cpp
    PlayerHistory.cpp
)

qt_add_qml_module(appDesktop_GUI_QML
    URI Desktop_GUI_QML
    VERSION 1.0
    QML_FILES
        Main.qml
        QML_FILES HomePage.qml
        QML_FILES ResultsPage.qml
        QML_FILES MetricCard.qml
)

# Qt for iOS sets MACOSX_BUNDLE_GUI_IDENTIFIER automatically since Qt 6.1.
# If you are developing for iOS or macOS you should consider setting an
# explicit, fixed bundle identifier manually though.
set_target_properties(appDesktop_GUI_QML PROPERTIES
#    MACOSX_BUNDLE_GUI_IDENTIFIER com.example.appDesktop_GUI_QML
    MACOSX_BUNDLE_BUNDLE_VERSION ${PROJECT_VERSION}
    MACOSX_BUNDLE_SHORT_VERSION_STRING ${PROJECT_VERSION_MAJOR}.${PROJECT_VERSION_MINOR}
    MACOSX_BUNDLE TRUE
    WIN32_EXECUTABLE TRUE
)

target_link_libraries(appDesktop_GUI_QML
    PRIVATE Qt6::Quick
)

include(GNUInstallDirs)
install(TARGETS appDesktop_GUI_QML
    BUNDLE DESTINATION .
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)
