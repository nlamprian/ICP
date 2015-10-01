/*! \file registration.cpp
 *  \brief An example showcasing the use of the `ICP` class.
 *  \details Estimates the transformation between two point clouds, registers 
 *           one of them to the other and displays the result with OpenGL.
 *  \note To change the configuration of the `%ICP` class, alter the `RC` and `WC` variables.
 *  \note **Command line arguments**:
 *  \note `<name_1>`: name of the binary file for the first point cloud.
 *  \note `<name_2>`: name of the binary file for the second point cloud.
 *  \note **Usage**:
 *  \note `./bin/icp_registration pcA pcB`
 *  \note loads the point clouds in `../data/pcA.bin` and `../data/pcB.bin`
 *  \note `./bin/icp_registration pc`
 *  \note loads the point clouds in `../data/pc_1.bin` and `../data/pc_2.bin`
 *  \note `./bin/icp_registration`
 *  \note loads the point clouds in `../data/kg_pc8d_1.bin` and `../data/kg_pc8d_2.bin`
 *  \author Nick Lamprianidis
 *  \version 1.1.0
 *  \date 2015
 *  \copyright The MIT License (MIT)
 *  \par
 *  Copyright (c) 2015 Nick Lamprianidis
 *  \par
 *  Permission is hereby granted, free of charge, to any person obtaining a copy
 *  of this software and associated documentation files (the "Software"), to deal
 *  in the Software without restriction, including without limitation the rights
 *  to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 *  copies of the Software, and to permit persons to whom the Software is
 *  furnished to do so, subject to the following conditions:
 *  \par
 *  The above copyright notice and this permission notice shall be included in
 *  all copies or substantial portions of the Software.
 *  \par
 *  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 *  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 *  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 *  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 *  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 *  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 *  THE SOFTWARE.
 */

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <GL/glew.h>  // Add before CLUtils.hpp
#include <CLUtils.hpp>
#include <ocl_icp_reg.hpp>

#if defined(__APPLE__) || defined(__MACOSX)
#include <GLUT/glut.h>
#else
#include <GL/glut.h>
#endif


// Window parameters
const int gl_win_width = 640;
const int gl_win_height = 480;
int glWinId;

// Model parameters
int mouseX = -1, mouseY = -1;
float dx = 0.f, dy = 0.f;
float angleX = 0.f, angleY = 0.f;
float zoom = 1.f;

// OpenGL buffer parameters
GLuint glPC4DBuffer, glRGBABuffer;

// Point cloud parameters
static const int width = 640;
static const int height = 480;
static const int n = width * height;
std::vector<cl_float8> pc8d1 (n), pc8d2 (n);

// OpenCL paramaters
const cl_algo::ICP::ICPStepConfigT RC = cl_algo::ICP::ICPStepConfigT::POWER_METHOD;
const cl_algo::ICP::ICPStepConfigW WC = cl_algo::ICP::ICPStepConfigW::WEIGHTED;
ICPReg<RC, WC> *icp;


/*! \brief Display callback for the window. */
void drawGLScene ()
{
    glClear (GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    
    // glPointSize(2.f);

    glBindBuffer (GL_ARRAY_BUFFER, glPC4DBuffer);
    glVertexPointer (4, GL_FLOAT, 0, NULL);
    glEnableClientState (GL_VERTEX_ARRAY);
    
    glBindBuffer (GL_ARRAY_BUFFER, glRGBABuffer);
    glColorPointer (4, GL_FLOAT, 0, NULL);
    glEnableClientState (GL_COLOR_ARRAY);
    
    glDrawArrays (GL_POINTS, 0, 2 * width * height);

    glDisableClientState (GL_VERTEX_ARRAY);
    glDisableClientState (GL_COLOR_ARRAY);
    glBindBuffer (GL_ARRAY_BUFFER, 0);

    // Draw the world coordinate frame
    glLineWidth (2.f);
    glBegin (GL_LINES);
    glColor3ub (255, 0, 0);
    glVertex3i (  0, 0, 0);
    glVertex3i ( 50, 0, 0);

    glColor3ub (0, 255, 0);
    glVertex3i (0,   0, 0);
    glVertex3i (0,  50, 0);

    glColor3ub (0, 0, 255);
    glVertex3i (0, 0,   0);
    glVertex3i (0, 0,  50);
    glEnd ();

    // Position the camera
    glMatrixMode (GL_MODELVIEW);
    glLoadIdentity ();
    glScalef (zoom, zoom, 1);
    gluLookAt ( -7*angleX, -7*angleY, -1000.0,
                      0.0,       0.0,  2000.0,
                      0.0,      -1.0,     0.0 );
    glTranslatef (dx, dy, 0.f);

    glutSwapBuffers ();
}


/*! \brief Idle callback for the window. */
void idleGLScene ()
{
    glutPostRedisplay ();
}


/*! \brief Reshape callback for the window. */
void resizeGLScene (int width, int height)
{
    glViewport (0, 0, width, height);
    glMatrixMode (GL_PROJECTION);
    glLoadIdentity ();
    gluPerspective (70.0, width / (float) height, 900.0, 11000.0);
    glMatrixMode (GL_MODELVIEW);
}


/*! \brief Keyboard callback for the window. */
void keyPressed (unsigned char key, int x, int y)
{
    switch (key)
    {
        case 0x1B:  // ESC
        case  'Q':
        case  'q':
            glutDestroyWindow (glWinId);
            break;
        case 'T':
        case 't':
            icp->registerPC ();
            break;
        case 'R':
        case 'r':
            icp->init (pc8d1, pc8d2);
            break;
    }
}


/*! \brief Arrow key callback for the window. */
void arrowPressed (int key, int x, int y)
{
    switch (key)
    {
        case GLUT_KEY_RIGHT:
            dx -= 200;
            break;
        case GLUT_KEY_LEFT:
            dx += 200;
            break;
        case GLUT_KEY_DOWN:
            dy -= 200;
            break;
        case GLUT_KEY_UP:
            dy += 200;
            break;
    }
}


/*! \brief Mouse callback for the window. */
void mouseMoved (int x, int y)
{
    if (mouseX >= 0 && mouseY >= 0)
    {
        angleX += x - mouseX;
        angleY += y - mouseY;
    }

    mouseX = x;
    mouseY = y;
}


/*! \brief Mouse button callback for the window. */
void mouseButtonPressed (int button, int state, int x, int y)
{
    if (state == GLUT_DOWN)
    {
        switch (button)
        {
            case GLUT_LEFT_BUTTON:
                mouseX = x;
                mouseY = y;
                break;
            case 3:  // Scroll Up
                zoom *= 1.2f;
                break;
            case 4:  // Scroll Down
                zoom /= 1.2f;
                break;
        }
    }
    else if (state == GLUT_UP && button == GLUT_LEFT_BUTTON)
    {
        mouseX = -1;
        mouseY = -1;
    }
}


/*! \brief Initializes GLUT. */
void initGL (int argc, char **argv)
{
    glutInit (&argc, argv);
    glutInitDisplayMode (GLUT_RGBA | GLUT_DOUBLE | GLUT_ALPHA);
    glutInitWindowSize (gl_win_width, gl_win_height);
    glutInitWindowPosition ((glutGet (GLUT_SCREEN_WIDTH) - gl_win_width) / 2,
                            (glutGet (GLUT_SCREEN_HEIGHT) - gl_win_height) / 2 - 70);
    glWinId = glutCreateWindow ("ICP Registration");

    glutDisplayFunc (&drawGLScene);
    glutIdleFunc (&idleGLScene);
    glutReshapeFunc (&resizeGLScene);
    glutKeyboardFunc (&keyPressed);
    glutSpecialFunc (&arrowPressed);
    glutMotionFunc (&mouseMoved);
    glutMouseFunc (&mouseButtonPressed);

    glewInit ();

    glClearColor (0.65f, 0.65f, 0.65f, 1.f);
    glEnable (GL_BLEND);
    glBlendFunc (GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glEnable (GL_ALPHA_TEST);
    glAlphaFunc (GL_GREATER, 0.f);
    glEnable (GL_DEPTH_TEST);
    glShadeModel (GL_SMOOTH);
}


/*! \brief Displays the available controls. */
void printInfo ()
{
    std::cout << "\nAvailable Controls:\n";
    std::cout << "===================\n";
    std::cout << " Perform ICP Registration :  T\n";
    std::cout << " Reset Transformation     :  R\n";
    std::cout << " Rotate                   :  Mouse Left Button\n";
    std::cout << " Zoom In/Out              :  Mouse Wheel\n";
    std::cout << " Quit                     :  Q or Esc\n\n";
}


/*! \brief Reads in a binary file.
 *  
 *  \param[in] path path to the file.
 *  \param[out] data array that receives the data.
 *  \param[in] n number of bytes to read.
 */
void fread (const char *path, char *data, size_t n)
{
    std::ifstream f (path, std::ios::binary);
    f.read (data, n);
    f.close ();
}


/*! \brief Configures parameters.
 *  
 *  \param[in] argc command line argument count
 *  \param[in] argv command line arguments
 */
void configure (int argc, char **argv)
{
    std::string filename1, filename2;

    if (argc == 1)
    {
        filename1 = std::string ("../data/kg_pc8d_1.bin");
        filename2 = std::string ("../data/kg_pc8d_2.bin");
    }
    else if (argc == 2)
    {
        std::string name (argv[1]);
        std::ostringstream filename;

        filename << "../data/" << name << "_1.bin";
        filename1 = filename.str ();
        
        filename.str (std::string ());
        filename.clear ();
        filename << "../data/" << name << "_2.bin";
        filename2 = filename.str ();
    }
    else
    {
        std::ostringstream filename;

        filename << "../data/" << argv[1] << ".bin";
        filename1 = filename.str ();
        
        filename.str (std::string ());
        filename.clear ();
        filename << "../data/" << argv[2] << ".bin";
        filename2 = filename.str ();
    }

    
    std::cout << "Loading 1st point cloud from " << filename1 << std::endl;
    fread (filename1.c_str (), (char *) pc8d1.data (), n * sizeof (cl_float8));
    std::cout << "Loading 2nd point cloud from " << filename2 << std::endl;
    fread (filename2.c_str (), (char *) pc8d2.data (), n * sizeof (cl_float8));
}


int main (int argc, char **argv)
{
    try
    {
        printInfo ();

        configure (argc, argv);

        initGL (argc, argv);

        // The OpenCL environment must be created after the OpenGL environment 
        // has been initialized and before OpenGL starts rendering
        icp = new ICPReg<RC, WC> (&glPC4DBuffer, &glRGBABuffer);
        icp->init (pc8d1, pc8d2);

        glutMainLoop ();

        delete icp;

        return 0;
    }
    catch (const cl::Error &error)
    {
        std::cerr << error.what ()
                  << " (" << clutils::getOpenCLErrorCodeString (error.err ()) 
                  << ")"  << std::endl;
    }
    exit (EXIT_FAILURE);
}
