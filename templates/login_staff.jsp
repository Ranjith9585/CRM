<%@ page contentType="text/html; charset=iso-8859-1" language="java" import="java.sql.*" errorPage="" %>
<%@ include file="dbconnect.jsp" %>
<!DOCTYPE html>
<html lang="en">
<head>

	<title>EDU Scanner</title>
	<!--

    Template 2106 Soft Landing

	http://www.tooplate.com/view/2106-soft-landing

    -->
     <meta charset="UTF-8">
     <meta http-equiv="X-UA-Compatible" content="IE=Edge">
     <meta name="description" content="">
     <meta name="keywords" content="">
     <meta name="team" content="">
     <meta name="viewport" content="width=device-width, initial-scale=1, maximum-scale=1">

     <link rel="stylesheet" href="css/bootstrap.min.css">
     <link rel="stylesheet" href="css/owl.carousel.css">
     <link rel="stylesheet" href="css/owl.theme.default.min.css">
     <link rel="stylesheet" href="css/font-awesome.min.css">

     <!-- MAIN CSS -->
     <link rel="stylesheet" href="css/tooplate-style.css">

</head>
<body>
<%
String msg="";

try
{
String btn=request.getParameter("btn");
String un=request.getParameter("uname");
String pw=request.getParameter("pass");
	if(btn.equals("Login"))
	{
		
		String qry="select * from admin where username='"+un+"' && password='"+pw+"'";
                ResultSet rs=stmt.executeQuery(qry);
			if(rs.next())
			{
			session.setAttribute("username",un);
			response.sendRedirect("admin.jsp");
			}
			else
			{
			msg="Invalid User!";
			}
	}	

}
catch(Exception e)
{
}
%>
     <!-- PRE LOADER -->
     <section class="preloader">
          <div class="spinner">

               <span class="spinner-rotate"></span>
               
          </div>
     </section>


     <!-- MENU -->
     <section class="navbar custom-navbar navbar-fixed-top" role="navigation">
          <div class="container">

               <div class="navbar-header">
                    <button class="navbar-toggle" data-toggle="collapse" data-target=".navbar-collapse">
                         <span class="icon icon-bar"></span>
                         <span class="icon icon-bar"></span>
                         <span class="icon icon-bar"></span>
                    </button>

                    <!-- lOGO TEXT HERE -->
                    <a href="" class="navbar-brand">EDU Scanner</a>
               </div>

               <!-- MENU LINKS -->
               <div class="collapse navbar-collapse">
                    <ul class="nav navbar-nav">
                       <li><a href="index.jsp" class="smoothScroll">Home</a></li>
						 <li><a href="login.jsp" class="smoothScroll">Login</a></li>
                    </ul>
                    <ul class="nav navbar-nav navbar-right">
                      
                    </ul>
               </div>

          </div>
     </section>

<!-- FEATURE -->
     <section id="home" data-stellar-background-ratio="0.5">
          <div class="overlay"></div>
          <div class="container">
               <div class="row">

                 

               </div>
          </div>
     </section>
	 
 <div class="row">
  
  			<div class="col-lg-3">
			
				<!-- A grey horizontal navbar that becomes vertical on small screens -->
			</div>
			
			
            <div class="col-lg-6">
              <div class="card">
                <div class="card-header d-flex align-items-center">
                  <h2 class="h5 display display">
                    <h3>Admin Login</h3>
                  </h2>
                </div>
                <div class="card-block">
                  <p></p>
                  <form name="name" method="post">
                   
					<div class="form-group">
                      <label>Username</label>
                      <input type="text" name="uname" placeholder="Username" class="form-control" required>
                    </div>
                    <div class="form-group">       
                      <label>Password</label>
                      <input type="password" name="pass" placeholder="Password" class="form-control" required>
                    </div>
                    <div class="form-group">       
                      <input type="submit" name="btn" value="Login" class="btn btn-primary">
                    </div>
					
                  </form>
				  <%
				  if(msg!="")
				  {
				  out.print(msg);
				  }
				  %>
                </div>
              </div>
    </div>
	
	
	<div class="col-lg-3">
			
				<!-- A grey horizontal navbar that becomes vertical on small screens -->
			</div>

</div>


   

     <!-- FOOTER -->
     <footer id="footer" data-stellar-background-ratio="0.5">
          <div class="container">
               <div class="row">

                    <div class="copyright-text col-md-12 col-sm-12">
                         <div class="col-md-6 col-sm-6">
                              <p>EDU Scanner </p>
                         </div>

                         <div class="col-md-6 col-sm-6">
                              <ul class="social-icon">
                                  
                              </ul>
                         </div>
                    </div>

               </div>
          </div>
     </footer>


     <!-- SCRIPTS -->
     <script src="js/jquery.js"></script>
     <script src="js/bootstrap.min.js"></script>
     <script src="js/jquery.stellar.min.js"></script>
     <script src="js/owl.carousel.min.js"></script>
     <script src="js/smoothscroll.js"></script>
     <script src="js/custom.js"></script>

</body>
</html>